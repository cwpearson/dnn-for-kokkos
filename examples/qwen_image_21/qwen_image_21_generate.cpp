// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 text-to-image driver: true-pipeline phase order
// (encode -> release+fence -> DiT sample -> release+fence -> VAE decode)
// with FlowEuler sampling and CFG.
//
// TEXT-TO-IMAGE ONLY: this driver conditions on text alone. Qwen-Image-Edit
// (reference-image conditioning) is out of scope and is rejected nowhere
// here simply because no such flag exists; do not add one in this TU.
// One-shot editing lives next door in qwen_image_21_edit.cpp (shared
// qwen_image_21_common.hpp implementation).
//
// Backend selection is a compile-time compatibility shim: when the
// true-pipeline headers are present in models/qwen_image_21/
// (KokkosDNN_QwenImage21Dit.hpp, KokkosDNN_QwenImage21Encoder.hpp,
// KokkosDNN_QwenImage21Vae.hpp, plus geometry in
// KokkosDNN_QwenImage21Config.hpp) they are picked up via __has_include and
// the corresponding phase runs the real model (encode: raw-text BPE +
// EncoderModel forward; decode: runtime-affine denormalize +
// VaeModel::decode to RGBA); otherwise the phase falls
// back to the placeholder Qwen21:: geometry / file / null path with a
// logged reason, so the tree always builds. The BPE tables come from
// models/moss_image/KokkosDNN_QwenTokenizer.hpp (probed, same shim).
// QWEN21_DRIVER_HAVE_{OPS,DIT,ENCODER,VAE,TOKENIZER} gate every use (see
// the shim below). Status: Config geometry + the DiT are usable when
// present (the driver samples with the real DiT, including --tiny on
// random weights); encode runs the in-process raw-text BPE +
// EncoderModel forward when --tokenizer + --text-encoder are given (else
// --prompt-embeds-in file or null), and decode runs the real VAE decode
// to RGBA by default when --vae-weights are given (else a grayscale
// preview; pass --no-vae-decode for the preview), each fallback with a
// logged reason.
//
// Interactive loop (default): without --prompt on a TTY (and without the
// file/parity handoff flags) the driver asks for a prompt, asks for a
// filename, generates, and repeats until EOF or quit/exit. Each iteration
// prints the current pixel HxW (and latent HxW) on the line before the
// prompt, offers the previous prompt for editing (UP/DN move through the
// in-memory prompt history), accepts /w N and /h N to set the pixel
// width/height going forward (rounded to the nearest patch-stride multiple,
// see --height), and treats ESC at the filename step as back-to-prompt
// (ESC re-asks at the prompt step). Giving --prompt
// (or any of --prompt-embeds-in/--text-weights, --init-latents-in,
// --dump-steps, --decode-latents-in, or non-TTY stdin) runs ONE generation
// and exits. The per-iteration seed is --seed + iteration; an empty
// filename auto-numbers qwen_image_21_out_<n>.png.
//
//   qwen_image_21_generate [--tiny] [--weights transformer_dir]
//                          [--prompt "text"]   (one-off; omit to loop)
//                          [--tokenizer tok_dir] [--text-encoder enc_dir]
//                          [--text-weights feats.f32]
//                          [--prompt-embeds-in feats.f32]
//                          [--height H] [--width W]   (output pixels;
//                           each rounds to the nearest 16*patch multiple)
//                          [--steps 8] [--cfg 1.0] [--shift 3.0]
//                          [--sigmas driver|turbo]
//                          [--seed 42] [--init-latents-in lat.f32]
//                          [--vae-weights vae.safetensors [--no-vae-decode]]
//                          [--decode-latents-in lat.f32
//                           [--decode-latents-layout chw|token]
//                           [--decode-out-f32 file]]
//                          [--out img.png]
//                          [--no-staged] [--dump-steps DIR]
//// Text conditioning precedence: in-process Qwen forward (--tokenizer dir
// with tokenizer.json + --text-encoder weights, raw T2I template, no chat
// template) -> --prompt-embeds-in file (alias
// --text-weights) -> null (zeros) with the prompt logged. --tokenizer /
// --text-encoder must be given together. --tiny skips the Qwen forward.
// --vae-weights is validated whenever given; VAE decode runs by default
// (latents denormalized by the runtime vae/config.json affine, then
// QwenImage21VaeModel::decode to RGBA; pass --no-vae-decode for the
// channel-mean grayscale preview).
//
// Staged execution (24GB fit, default): encode -> releases the encoder
// weights (the MiB-scale embeds stay resident on the device) ->
// Kokkos::fence() -> loads the DiT -> samples straight from the
// device-resident embeds -> releases the DiT -> Kokkos::fence() -> decodes,
// so the 17.5GB encoder, the 14.2GB DiT, and the VAE are never resident at
// once. Pass --no-staged for the legacy single-phase run. No disk round-trip
// and no second process: the embeds are separate
// (MiB) allocations from the (GiB) encoder weights. In the interactive loop
// the encoder and transformer additionally live in HOST memory (loaded once
// before the first prompt) and are copied host->device per iteration, so
// staging (default) keeps the same one-component-resident property every
// iteration
// without re-reading the ~32 GB checkpoint from disk. Each phase logs its
// resident estimate (weights bytes + activation estimate).
//
// --tiny runs a random-weights smoke model on 4x4 latents, 2 steps
// (no downloads).
//
// --dump-steps DIR writes golden-parity tensors as raw little-endian float32
// (driver layout: latents [C,H,W] row-major C-major lat[(c*H+h)*W+w])
// plus manifest.json: initial_latents.f32, then
// stepNN_noise_pred.f32 / stepNN_latents.f32 per step (with
// stepNN_noise_pred_cond.f32 / _uncond.f32 when the CFG branch runs), each
// logged with its f64 sum. Off by default: the sample loop only takes the
// extra device->host copies when the flag is set.
#include "qwen_image_21_common.hpp"
#include "KokkosDNN_ExampleReadline.hpp"

void usage();
template <class CfgT>
int run_interactive_loop(const CfgT& cfg, const struct LoopFixedOpts& o);

// ---- Interactive generation loop (default) --------------------------------
// When --prompt is absent (and no parity/file handoff flags are given) the
// driver runs as a REPL on a TTY: it asks for a prompt, asks for an output
// filename, generates, and repeats indefinitely until EOF or quit/exit.
// The encoder and transformer weights are loaded ONCE into host memory
// (HostSpace mirrors) before the first prompt; every iteration copies the
// needed component host->device, runs the phase, and releases the device
// copy -- so staging (default) keeps the same one-component-resident property
// per iteration without re-reading the ~32 GB checkpoint from disk.
// File/parity modes (--prompt-embeds-in/--text-weights, --init-latents-in,
// --dump-steps, --decode-latents-in) and non-TTY stdin force the legacy
// one-off run. In the loop the latent seed is base --seed + iteration, so
// successive images differ even for the same prompt.

inline bool stdin_is_tty() { return ::isatty(STDIN_FILENO) != 0; }

inline bool is_quit_word(std::string_view s) {
  return s == "quit" || s == "exit" || s == ":q";
}

// Trim ASCII blanks from both ends so "  /w 128  " still parses as a size
// command and whitespace-only lines re-ask like empty ones.
inline std::string_view trim_ws(std::string_view s) {
  const size_t b = s.find_first_not_of(" \t");
  if (b == std::string_view::npos) return {};
  const size_t e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

// Interactive "/w N" / "/h N": set the output pixel width/height going
// forward (rounded to the nearest patch-stride multiple at generation time).
// Returns true when the line was a size command (handled: the caller
// re-prompts without generating). Anything else starting with '/' falls
// through as a normal prompt, so "/where ..." is never hijacked.
inline bool try_size_command(std::string_view line, int& Hpx, int& Wpx,
                             bool tiny) {
  const std::string_view t = trim_ws(line);
  if (t.empty() || t[0] != '/') return false;
  const char cmd = (t.size() > 1) ? t[1] : '\0';
  if (cmd != 'w' && cmd != 'h') return false;
  // Require end/blank/digit after the letter so "/where ..." stays a prompt.
  if (t.size() > 2 && t[2] != ' ' && t[2] != '\t' &&
      (t[2] < '0' || t[2] > '9'))
    return false;
  const std::string_view rest = trim_ws(t.substr(2));
  if (rest.empty()) {
    std::printf("[size] %dx%d px\n", Hpx, Wpx);
    std::fflush(stdout);
    return true;
  }
  int val = 0;
  try {
    size_t pos = 0;
    val = std::stoi(std::string(rest), &pos);
    if (pos != rest.size()) throw std::invalid_argument("trailing text");
  } catch (const std::exception&) {
    std::printf("[size] usage: /w N | /h N (pixels; got \"%.*s\")\n",
                static_cast<int>(rest.size()), rest.data());
    std::fflush(stdout);
    return true;
  }
  if (tiny) {
    std::printf("[size] --tiny uses fixed 4x4 latents; ignoring /%c %d\n", cmd,
                val);
    std::fflush(stdout);
    return true;
  }
  const char* what = (cmd == 'w') ? "width" : "height";
  if (val <= 0) {
    std::printf("[size] %s pixels must be positive (got %d)\n", what, val);
    std::fflush(stdout);
    return true;
  }
  if (cmd == 'w') {
    Wpx = val;
  } else {
    Hpx = val;
  }
  std::printf("[size] now %dx%d px (rounded to the patch stride when "
              "generating)\n",
              Hpx, Wpx);
  std::fflush(stdout);
  return true;
}

#ifdef QWEN21_DRIVER_HAVE_DIT
// Host<->device Dit weight copy (both directions: instantiate with
// DstExec/SrcExec as needed). Both sides must be allocated from the same
// config; extents are checked by deep_copy.
template <class DstExec, class SrcExec>
void copy_dit_weights(
    KokkosDNN::QwenImage21Dit::QwenImage21DitWeights<DstExec, DScalar>& dst,
    const KokkosDNN::QwenImage21Dit::QwenImage21DitWeights<SrcExec, DScalar>&
        src) {
  if (dst.blocks.size() != src.blocks.size())
    throw std::runtime_error("copy_dit_weights: block count mismatch");
  Kokkos::deep_copy(dst.txt_norm, src.txt_norm);
  Kokkos::deep_copy(dst.txt_in_W, src.txt_in_W);
  Kokkos::deep_copy(dst.txt_out_W, src.txt_out_W);
  Kokkos::deep_copy(dst.img_in_W, src.img_in_W);
  Kokkos::deep_copy(dst.time1_W, src.time1_W);
  Kokkos::deep_copy(dst.time2_W, src.time2_W);
  Kokkos::deep_copy(dst.mod_W, src.mod_W);
  Kokkos::deep_copy(dst.norm_out_W, src.norm_out_W);
  Kokkos::deep_copy(dst.proj_out_W, src.proj_out_W);
  for (size_t i = 0; i < dst.blocks.size(); ++i) {
    auto& d = dst.blocks[i];
    const auto& s = src.blocks[i];
    Kokkos::deep_copy(d.to_q, s.to_q);
    Kokkos::deep_copy(d.to_k, s.to_k);
    Kokkos::deep_copy(d.to_v, s.to_v);
    Kokkos::deep_copy(d.to_o, s.to_o);
    Kokkos::deep_copy(d.norm_q, s.norm_q);
    Kokkos::deep_copy(d.norm_k, s.norm_k);
    Kokkos::deep_copy(d.proj, s.proj);
    Kokkos::deep_copy(d.gate, s.gate);
    Kokkos::deep_copy(d.out, s.out);
  }
  Kokkos::fence();
}
#endif

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
// Host<->device encoder weight copy (bhalf_t, matching run_qwen's
// EncoderModel<Exec, bhalf_t>).
template <class DstExec, class SrcExec>
void copy_enc_weights(
    KokkosDNN::QwenImage21Encoder::QwenImage21EncoderWeights<
        DstExec, Kokkos::Experimental::bhalf_t>& dst,
    const KokkosDNN::QwenImage21Encoder::QwenImage21EncoderWeights<
        SrcExec, Kokkos::Experimental::bhalf_t>& src) {
  if (dst.layers.size() != src.layers.size())
    throw std::runtime_error("copy_enc_weights: layer count mismatch");
  Kokkos::deep_copy(dst.embed, src.embed);
  Kokkos::deep_copy(dst.final_norm, src.final_norm);
  for (size_t i = 0; i < dst.layers.size(); ++i) {
    auto& d = dst.layers[i];
    const auto& s = src.layers[i];
    Kokkos::deep_copy(d.q_W, s.q_W);
    Kokkos::deep_copy(d.k_W, s.k_W);
    Kokkos::deep_copy(d.v_W, s.v_W);
    Kokkos::deep_copy(d.o_W, s.o_W);
    Kokkos::deep_copy(d.qn, s.qn);
    Kokkos::deep_copy(d.kn, s.kn);
    Kokkos::deep_copy(d.in_norm, s.in_norm);
    Kokkos::deep_copy(d.post_norm, s.post_norm);
    Kokkos::deep_copy(d.gate_W, s.gate_W);
    Kokkos::deep_copy(d.up_W, s.up_W);
    Kokkos::deep_copy(d.down_W, s.down_W);
  }
  Kokkos::fence();
}
#endif

// Fixed generation options shared by every loop iteration (prompt, output
// path, and per-iteration seed vary).
struct LoopFixedOpts {
  std::string weights = "transformer";
  std::string tokenizer_dir;
  std::string text_encoder;
  int H = 1024, W = 1024;  // requested output pixels (rounded at generation)
  int steps_eff = 8;
  float guidance = 1.0f, flow_shift = 3.0f;
  std::string sigmas_mode = "driver";
  uint64_t seed = 42;
  std::string vae_weights;
  bool vae_decode = true;
  std::string out_default = "qwen_image_21_out.png";
  bool staged = true;
  bool tiny = false;
};

void usage() {
  std::printf(
      "usage: qwen_image_21_generate [--tiny] [--weights transformer_dir]\n"
      "                    [--prompt \"text\"]\n"
      "                    [--tokenizer tok_dir] [--text-encoder enc_dir]\n"
      "                    [--text-weights feats.f32]\n"
      "                    [--prompt-embeds-in feats.f32]\n"
      "                    [--height H] [--width W]\n"
      "                    [--steps N] [--cfg G] [--shift S]\n"
      "                    [--sigmas driver|turbo]\n"
      "                    [--seed N] [--init-latents-in lat.f32]\n"
      "                    [--vae-weights vae.safetensors [--no-vae-decode]]\n"
      "                    [--decode-latents-in lat.f32\n"
      "                     [--decode-latents-layout chw|token]\n"
      "                     [--decode-out-f32 file]]\n"
      "                    [--out img.png]\n"
      "                    [--no-staged] [--dump-steps DIR]\n"
      "\n"
      "  --tiny         random-weights smoke model, 4x4 latents, 2 steps\n"
      "                 (no files)\n"
      "  --weights      DiT transformer checkpoint: transformer/ shard dir\n"
      "                 or single .safetensors (default: transformer)\n"
      "  --prompt       caption text (Qwen-encoded in-process with\n"
      "                 --tokenizer + --text-encoder, else needs\n"
      "                 --text-weights, else null). Giving --prompt runs ONE\n"
      "                 generation and exits; without it (TTY stdin, no\n"
      "                 --prompt-embeds-in/--text-weights/--init-latents-in/\n"
      "                 --dump-steps/--decode-latents-in) the driver loops:\n"
      "                 it asks for a prompt, asks for a filename, generates,\n"
      "                 and repeats until EOF or quit/exit (empty prompt\n"
      "                 re-asks; empty filename auto-numbers). The per-iteration\n"
      "                 seed is --seed + iteration. In the loop /w N and /h N\n"
      "                 set the pixel width/height going forward (rounded to\n"
      "                 the patch stride at generation time), and the\n"
      "                 previous prompt is offered for editing (UP/DN move\n"
      "                 through the in-memory prompt history). ESC at output>\n"
      "                 goes back to prompt> (previous prompt pre-filled);\n"
      "                 ESC at prompt> re-asks.\n"
      "  --tokenizer    dir holding tokenizer.json;\n"
      "                 with --text-encoder runs the in-process Qwen forward\n"
      "                 (raw T2I template, no chat template)\n"
      "  --text-weights raw float32 LE caption features [L,Cc]\n"
      "                 (L <= maxlen, Cc == caption dim)\n"
      "  --text-encoder Qwen text-encoder weights: text_encoder/ shard dir\n"
      "                 or single .safetensors\n"
      "  --prompt-embeds-in alias of --text-weights (wins if both differ)\n"
      "  --staged       (default ON) encode -> release encoder weights (MiB\n"
      "                 embeds stay resident on device) -> fence -> load DiT\n"
      "                 -> sample from device embeds -> release DiT -> fence\n"
      "                 -> decode (one component resident at a time, no disk\n"
      "                 round-trip). Pass --no-staged for the legacy\n"
      "                 single-phase run\n"
      "  --height/--width output pixels H/W (default 1024; each rounds to\n"
      "                 the nearest 16*patch px multiple, so the latent grid\n"
      "                 pixels/16 tiles the DiT patch; true path stride 16px,\n"
      "                 placeholder stride 32px)\n"
      "  --steps        Euler steps (default 8 Turbo; --tiny forces 2\n"
      "                 unless --steps is given)\n"
      "  --cfg/--guidance CFG scale (!=1 takes the CFG branch; default 1.0)\n"
      "  --shift/--flow-shift flow-matching sigma shift (default 3.0;\n"
      "                 ignored under --sigmas turbo)\n"
      "  --sigmas       'driver' (default): nearest-grid schedule from\n"
      "                 --steps/--shift; 'turbo': checkpoint sample_sigmas\n"
      "                 verbatim (mu-shifted already, 8 steps, shift=1.0)\n"
      "  --seed         latent init seed (default 42; ignored when\n"
      "                 --init-latents-in is given)\n"
      "  --init-latents-in raw float32 LE initial latents [C,H,W]\n"
      "                 row-major, flattened C-major lat[(c*H+h)*W+w]\n"
      "                 (C*H*W floats, PRE-denormalization latent noise:\n"
      "                 no mean/std applied); replaces the mt19937 sample\n"
      "  --vae-weights  VAE checkpoint (validated whenever given)\n"
      "  --vae-decode   (default ON) run the real VAE decode (denormalize by\n"
      "                 the runtime vae/config.json affine, decode to RGBA)\n"
      "                 when the VAE header + weights are present, else\n"
      "                 grayscale preview. Pass --no-vae-decode for the preview\n"
      "  --decode-latents-in raw fp32 LE latents to decode directly (skips\n"
      "                 encode, DiT load and sampling; needs --vae-weights;\n"
      "                 --height/--width are the output pixels)\n"
      "  --decode-latents-layout 'chw' (default: lat[(c*H+h)*W+w]) or\n"
      "                 'token' (torch packed [H*W,C], golden final latents)\n"
      "  --decode-out-f32 also write the post-clamp RGBA decode as raw f32\n"
      "  --out          PNG with KokkosDNN_ENABLE_IMAGE=ON, else PPM\n"
      "  --dump-steps   write golden-parity tensors (raw float32 LE + f64\n"
      "                 sums + manifest.json) into DIR: initial_latents, then\n"
      "                 stepNN_noise_pred / stepNN_latents per Euler step;\n"
      "                 off by default (no extra device->host copies)\n");
}

// ---- Interactive loop driver (both backends) -------------------------------
// CfgT is KokkosDNN::QwenImage21::Config (true pipeline) or QW::Config
// (placeholder); be_* traits select geometry. Parity/file handoff modes are
// excluded by construction (main forces one-off when --prompt-embeds-in,
// --text-weights, --init-latents-in, --dump-steps, or --decode-latents-in is
// given), so this path never dumps and always samples fresh latents.
template <class CfgT>
int run_interactive_loop(const CfgT& cfg, const LoopFixedOpts& o) {
  // Mutable geometry: /w N and /h N update the requested pixels going
  // forward (rounded to the patch stride per iteration below). --tiny pins
  // 4x4 latents.
  const int patch = be_patch(cfg);
  int cur_Hpx = o.tiny ? 4 * kPixelScale : o.H;
  int cur_Wpx = o.tiny ? 4 * kPixelScale : o.W;
  const auto resolve_hw = [&](int req_h_px, int req_w_px) {
    if (o.tiny) {
      PixelGeometry g;
      g.px_h = g.px_w = 4 * kPixelScale;
      g.lat_h = g.lat_w = 4;
      return g;
    }
    return resolve_pixel_geometry(req_h_px, req_w_px, patch);
  };
  PixelGeometry cur_geo = resolve_hw(cur_Hpx, cur_Wpx);
  if (!o.tiny && (cur_geo.px_h != cur_Hpx || cur_geo.px_w != cur_Wpx))
    std::printf("[size] rounded %dx%d px -> %dx%d px (latent %dx%d, stride "
                "%dpx)\n",
                cur_Hpx, cur_Wpx, cur_geo.px_h, cur_geo.px_w, cur_geo.lat_h,
                cur_geo.lat_w, kPixelScale * patch);
  const size_t C = be_latent_channels(cfg);
  std::printf("[model %s: %s]\n", o.tiny ? "tiny_test" : "qwen-image-2.1",
              be_label(cfg));

  // VAE checkpoint validated once (decoded per iteration from disk; 0.7GB).
  // VAE decode is ON by default whenever --vae-weights are given.
  bool vae_decode = o.vae_decode;
  if (!o.vae_weights.empty()) {
    std::error_code ec0;
    const bool vae_is_dir = std::filesystem::is_directory(o.vae_weights, ec0);
    if (!vae_is_dir && !file_exists(o.vae_weights))
      throw std::runtime_error("vae weights file '" + o.vae_weights +
                               "' not found");
#ifdef QWEN21_DRIVER_HAVE_VAE
    std::printf("[vae %s] validated%s\n", o.vae_weights.c_str(),
                vae_decode ? "; decode enabled (default)"
                           : "; preview (pass --vae-decode to decode)");
#else
    std::printf("[vae %s] validated, but KokkosDNN_QwenImage21Vae.hpp is "
                "not in this build; writing a grayscale preview instead\n",
                o.vae_weights.c_str());
#endif
  } else if (vae_decode) {
    std::printf("[vae] VAE decode needs --vae-weights (else grayscale "
                "preview); disabling decode for this run\n");
    vae_decode = false;
  }

#ifdef QWEN21_DRIVER_HAVE_DIT
  // DiT weights live in host memory from here on; each iteration copies
  // host->device, samples, and releases the device copy.
  KokkosDNN::QwenImage21Dit::QwenImage21DitWeights<Kokkos::HostSpace, DScalar>
      dit_host;
  bool have_dit_host = false;
  if (!o.tiny) {
    const std::vector<std::string> shards =
        resolve_shard_list(o.weights, "--weights");
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> tmp(cfg);
    if (shards.size() > 1)
      std::printf("[loading %zu shards]\n", shards.size());
    else
      std::printf("[loading %s]\n", o.weights.c_str());
    try {
      tmp.weights().load_safetensors(shards);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string(e.what()) +
                               " (need Qwen-Image-2.1 weights or --tiny)");
    }
    dit_host.allocate(cfg);
    copy_dit_weights(dit_host, tmp.weights());
    have_dit_host = true;
    std::printf("[loop] DiT weights resident in host memory: %.1f GB "
                "(copied host->device each iteration%s)\n",
                (double)tmp.weights().resident_bytes() / 1e9,
                o.staged ? ", staged" : "");
  }
#endif

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
  // In-process Qwen inputs preloaded once: BPE tables + host encoder
  // weights. The per-iteration forward copies host->device into a fresh
  // device model and drops it when the embeds are staged.
  RawQwenTables loop_tabs;
  std::vector<int> loop_head_ids;
  size_t loop_drop = 0;
  KokkosDNN::QwenImage21Encoder::Config loop_ecfg;
  KokkosDNN::QwenImage21Encoder::QwenImage21EncoderWeights<
      Kokkos::HostSpace, Kokkos::Experimental::bhalf_t>
      enc_host;
  bool have_enc_host = false;
  if (!o.tiny && (!o.tokenizer_dir.empty() || !o.text_encoder.empty())) {
    if (o.tokenizer_dir.empty() || o.text_encoder.empty())
      throw std::runtime_error(
          "--tokenizer and --text-encoder must be given together for the "
          "in-process Qwen forward");
    namespace QELoop = KokkosDNN::QwenImage21Encoder;
    const std::string tok_json = o.tokenizer_dir + "/tokenizer.json";
    if (!file_exists(tok_json))
      throw std::runtime_error("--tokenizer '" + o.tokenizer_dir +
                               "' has no tokenizer.json (need the "
                               "Qwen-Image-2.1 processor/ dir)");
    loop_tabs = load_raw_qwen_tables(tok_json);
    const size_t max_len = be_max_seq(cfg);
    const std::string kUserPrefix = "<|im_start|>user\n";
    const std::string full_head = QELoop::qwen21_t2i_head();
    if (full_head.size() < kUserPrefix.size() ||
        full_head.compare(full_head.size() - kUserPrefix.size(),
                          kUserPrefix.size(), kUserPrefix) != 0)
      throw std::runtime_error(
          "in-process Qwen encode: T2I head lost its user-role prefix");
    const std::string sys_block =
        full_head.substr(0, full_head.size() - kUserPrefix.size());
    const std::vector<int> sys_ids =
        encode_qwen_raw(loop_tabs, sys_block, max_len);
    const std::vector<int> prefix_ids =
        encode_qwen_raw(loop_tabs, kUserPrefix, max_len);
    loop_head_ids = encode_qwen_raw(loop_tabs, full_head, max_len);
    if (sys_ids.empty() || prefix_ids.empty() || loop_head_ids.empty() ||
        loop_head_ids.size() != sys_ids.size() + prefix_ids.size() ||
        !std::equal(sys_ids.begin(), sys_ids.end(), loop_head_ids.begin()) ||
        !std::equal(prefix_ids.begin(), prefix_ids.end(),
                    loop_head_ids.begin() + (long)sys_ids.size()))
      throw std::runtime_error(
          "in-process Qwen encode: T2I head/prefix split mismatch");
    loop_drop = sys_ids.size();
    loop_ecfg.validate();
    if ((size_t)loop_ecfg.hidden_dim != be_caption_dim(cfg))
      throw std::runtime_error(
          "Qwen encoder hidden (" + std::to_string(loop_ecfg.hidden_dim) +
          ") != caption dim (" + std::to_string(be_caption_dim(cfg)) + ")");
    KokkosDNN::QwenImage21Encoder::QwenImage21EncoderModel<
        Exec, Kokkos::Experimental::bhalf_t>
        enc_tmp(loop_ecfg);
    const std::vector<std::string> eshards =
        resolve_shard_list(o.text_encoder, "--text-encoder");
    if (eshards.size() != 1)
      throw std::runtime_error(
          "--text-encoder must be the single-shard text_encoder/ "
          "model.safetensors (got " +
          std::to_string(eshards.size()) + " shards)");
    std::printf("[loading %s]\n", o.text_encoder.c_str());
    enc_tmp.weights().load_safetensors(eshards[0]);
    enc_host.allocate(loop_ecfg);
    copy_enc_weights(enc_host, enc_tmp.weights());
    have_enc_host = true;
    std::printf("[loop] Qwen encoder weights resident in host memory: %.1f "
                "GB (copied host->device each iteration)\n",
                (double)enc_tmp.weights().resident_bytes() / 1e9);
  }
  if (o.tiny && (!o.tokenizer_dir.empty() || !o.text_encoder.empty()))
    std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                "smoke model); using null embeddings\n");
#else
  if (!o.tiny && (!o.tokenizer_dir.empty() || !o.text_encoder.empty()))
    std::printf("[encode] --tokenizer/--text-encoder need "
                "KokkosDNN_QwenImage21Encoder.hpp + "
                "KokkosDNN_QwenTokenizer.hpp, not present in this build; "
                "using null embeddings\n");
#endif

  // Sigma schedule is prompt-independent: build once, reuse every iteration.
  std::vector<float> base_sigmas;
  if (o.sigmas_mode == "turbo") {
#ifdef QWEN21_DRIVER_HAVE_OPS
    for (double s : cfg.sample_sigmas)
      base_sigmas.push_back(static_cast<float>(s));
    base_sigmas.push_back(0.0f);
    std::printf("[schedule turbo] checkpoint sample_sigmas verbatim "
                "(mu-shifted already)\n");
#else
    throw std::runtime_error(
        "--sigmas turbo needs the true QwenImage21::Config "
        "(KokkosDNN_QwenImage21Config.hpp is not in this build)");
#endif
  } else {
    base_sigmas = build_sigmas_nearest_grid(o.steps_eff, o.flow_shift);
  }
  const int steps_run = static_cast<int>(base_sigmas.size()) - 1;

  std::printf(
      "[loop] interactive generation: prompt, then output filename, per "
      "iteration (EOF or quit/exit stops; empty prompt re-asks; empty "
      "filename auto-numbers qwen_image_21_out_<n>.png; seed = %llu + "
      "iteration)\n",
      (unsigned long long)o.seed);
  if (!o.tiny) {
    std::printf("[loop] /w N and /h N set the pixel width/height for later "
                "iterations; the previous prompt is offered for editing\n");
  } else if (KokkosDNN::Example::Readline::Available()) {
    std::printf("[loop] readline editing enabled (previous prompt offered "
                "for editing each time)\n");
  }
  std::fflush(stdout);
  size_t iter = 0;
  std::string prev_prompt;
  while (true) {
    if (!o.tiny) {
      // Current geometry on its own line, ahead of the prompt line.
      cur_geo = resolve_hw(cur_Hpx, cur_Wpx);
      std::printf("[size] %dx%d px (latent %dx%d)\n", cur_geo.px_h,
                  cur_geo.px_w, cur_geo.lat_h, cur_geo.lat_w);
      std::fflush(stdout);
    }
    std::string cur_prompt;
    const std::string* prefill =
        prev_prompt.empty() ? nullptr : &prev_prompt;
    bool prompt_esc = false;
    if (!KokkosDNN::Example::Readline::PromptLine("prompt> ", cur_prompt,
                                                 prefill,
                                                 /*add_to_history=*/true,
                                                 &prompt_esc))
      break;
    if (prompt_esc) continue;  // ESC re-asks with the same prefill
    if (try_size_command(cur_prompt, cur_Hpx, cur_Wpx, o.tiny)) continue;
    if (is_quit_word(trim_ws(cur_prompt))) break;
    if (trim_ws(cur_prompt).empty()) continue;
    prev_prompt = cur_prompt;
    std::string cur_out_raw;
    bool out_esc = false;
    if (!KokkosDNN::Example::Readline::PromptLine("output> ", cur_out_raw,
                                                 /*prefill=*/nullptr,
                                                 /*add_to_history=*/false,
                                                 &out_esc))
      break;
    if (out_esc) continue;  // ESC at output> goes back to prompt>
    if (is_quit_word(trim_ws(cur_out_raw))) continue;  // cancel this one
    const std::string cur_out =
        trim_ws(cur_out_raw).empty()
            ? ("qwen_image_21_out_" + std::to_string(iter) + ".png")
            : std::string(trim_ws(cur_out_raw));
    // Per-iteration geometry snapshot (/w /h set pixel requests; rounding to
    // the patch stride happens here, belt-and-braces, before allocating).
    const int H = cur_geo.lat_h, W = cur_geo.lat_w;
    const int pxH = cur_geo.px_h, pxW = cur_geo.px_w;
    if (!o.tiny && (cur_geo.px_h != cur_Hpx || cur_geo.px_w != cur_Wpx))
      std::printf("[size] rounded %dx%d px -> %dx%d px (latent %dx%d)\n",
                  cur_Hpx, cur_Wpx, pxH, pxW, H, W);
    const uint64_t cur_seed = o.seed + iter;
    std::printf("[iter %zu] seed=%llu out=%s\n", iter,
                (unsigned long long)cur_seed, cur_out.c_str());
    std::fflush(stdout);

    // ---- Phase 1: encode (device encoder is a per-iteration copy) ----
    EncoderStage enc;
    bool encoded = false;
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
    if (have_enc_host) {
      try {
        namespace QEIter = KokkosDNN::QwenImage21Encoder;
        const size_t max_len = be_max_seq(cfg);
        const std::vector<int> ids = encode_qwen_raw(
            loop_tabs, QEIter::qwen21_format_prompt_t2i(cur_prompt), max_len);
        if (loop_head_ids.empty() || ids.size() <= loop_head_ids.size())
          throw std::runtime_error(
              "in-process Qwen encode: empty conditioning sequence");
        KokkosDNN::QwenImage21Encoder::QwenImage21EncoderModel<
            Exec, Kokkos::Experimental::bhalf_t>
            enc_model(loop_ecfg);
        copy_enc_weights(enc_model.weights(), enc_host);
        Kokkos::View<int**, Kokkos::LayoutRight, Exec> toks("qwen21enc_toks",
                                                            1, ids.size());
        {
          auto hm = Kokkos::create_mirror_view(toks);
          for (size_t s = 0; s < ids.size(); ++s) hm(0, s) = ids[s];
          Kokkos::deep_copy(toks, hm);
        }
        Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight,
                      Exec>
            hid("qwen21enc_hid", 1, ids.size(),
                (size_t)loop_ecfg.hidden_dim);
        KokkosDNN::DNNHandle<Exec> enc_handle;
        enc_model.forward(enc_handle, toks, hid, (int)loop_drop);
        Kokkos::fence();
        auto hhid =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hid);
        const size_t L = ids.size() - loop_drop;
        const size_t Cc = be_caption_dim(cfg);
        enc.L = L;
        enc.Cc = Cc;
        enc.embeds.assign(L * Cc, 0.0f);
        enc.neg_embeds.assign(L * Cc, 0.0f);
        enc.source = "qwen";
        for (size_t t = 0; t < L; ++t)
          for (size_t h = 0; h < Cc; ++h)
            enc.embeds[t * Cc + h] = (float)hhid(0, loop_drop + t, h);
        std::printf("[caption qwen] %zux%zu features (host->device copy, "
                    "drop_idx=%zu)\n",
                    enc.L, enc.Cc, loop_drop);
        enc.stage_device();
        encoded = true;
        // enc_model goes out of scope here: the device encoder copy is
        // released while the host cache above survives for the next prompt.
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: in-process Qwen encode failed (%s); using "
                     "null embedding\n",
                     e.what());
      }
    }
#endif
    if (!encoded) {
      enc.L = be_max_seq(cfg);
      enc.Cc = be_caption_dim(cfg);
      enc.embeds.assign(enc.L * enc.Cc, 0.0f);
      enc.neg_embeds.assign(enc.L * enc.Cc, 0.0f);
      enc.source = "null";
      std::printf("[caption null] prompt \"%s\" has no features here; using "
                  "zeros (%zux%zu)\n",
                  cur_prompt.c_str(), enc.L, enc.Cc);
      enc.stage_device();
    }
    log_encode_memory(enc);
    if (o.staged) {
      Kokkos::fence();
      QW::print_phase_memory("encode-released", enc.device_bytes(), 0);
      std::printf("[staged] keeping %.1f MiB of embeds resident on device "
                  "across the DiT load (no disk round-trip)\n",
                  (double)enc.device_bytes() / (1024.0 * 1024.0));
    }

    // ---- Phase 2: DiT (device copy restored from the host cache) ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> dit_model(
        cfg);
    {
      auto& w = dit_model.weights();
      if (have_dit_host) {
        copy_dit_weights(w, dit_host);
        std::printf("[loading host->device %.1f GB]\n",
                    (double)w.resident_bytes() / 1e9);
      } else if (o.tiny) {
        w.random_init(o.seed);
        std::printf("[tiny random weights] layers=%d hidden=%d latent=%dx%d\n",
                    cfg.num_layers, cfg.hidden_dim, H, W);
      } else {
        throw std::runtime_error(
            "need Qwen-Image-2.1 weights or --tiny (unreachable: host load "
            "would have failed before the loop)");
      }
      QW::print_phase_memory("sample", w.resident_bytes(),
                             dit_model.scratch_bytes());
    }
#else
    QW::Weights<Exec, DScalar> w;
    if (o.tiny) {
      QW::random_init(cfg, w, o.seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      throw std::runtime_error(
          "weights file '" + o.weights +
          "' needs the true Qwen-Image-2.1 DiT "
          "(KokkosDNN_QwenImage21Dit.hpp is not in this build); pass --tiny "
          "for the random-weights smoke run. See README for the expected "
          "checkpoint layout.");
    }
    QW::print_phase_memory("sample", w.resident_bytes(), w.resident_bytes());
#endif
    std::printf("[budget] full-checkpoint projection: DiT %.1f GB + VAE %.3f "
                "GB + activations vs 24GB (see README fit table)\n",
                QW::kDitBudgetBytes / 1e9, QW::kVaeBudgetBytes / 1e9);

    std::vector<float> x_init =
        QW::sample_gaussian(C * (size_t)H * (size_t)W, cur_seed);
    std::printf("[sample] %dx%d px (latent %dx%d) steps=%d cfg=%.3g shift=%.3g "
                "seed=%llu\n",
                pxH, pxW, H, W, steps_run, (double)o.guidance, (double)o.flow_shift,
                (unsigned long long)cur_seed);
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_run,
        o.guidance, o.flow_shift, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        },
        &base_sigmas, StepDumpHook{});
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("[done] %.2fs total, %.2fs/step\n", secs,
                secs / (double)std::max(steps_run, 1));

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

    // Captions are no longer needed: drop host + device copies before decode.
    enc.release();
    // ---- Phase 3: decode (the DiT device copy is released first) ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    dit_model.weights().release();
    dit_model.release_scratch();
#else
    w.release();
#endif
    Kokkos::fence();
    QW::print_phase_memory("dit-released", 0, 0);
    size_t oH = 0, oW = 0;
    [[maybe_unused]] int oCh = 1;
    std::vector<unsigned char> pix;
    bool decoded = false;
    if (vae_decode) {
#ifdef QWEN21_DRIVER_HAVE_VAE
      try {
        decoded = try_vae_decode_rgba(lat, C, (size_t)H, (size_t)W,
                                      o.vae_weights, pix, oH, oW, oCh);
        std::printf("[decode] VAE RGBA %zux%zux%d ready for output\n", oH, oW,
                    oCh);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: real VAE decode failed (%s); writing a "
                     "grayscale preview instead\n",
                     e.what());
      }
#else
      std::printf("[vae] VAE decode requested (default) but "
                  "KokkosDNN_QwenImage21Vae.hpp is not in this build; writing "
                  "a grayscale preview instead (need the VAE worktree "
                  "header + --vae-weights)\n");
#endif
    }
    if (!decoded) {
      pix = render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
      QW::print_phase_memory("decode-preview", 0, 0);
    }
    write_output_image(cur_out, pix, oH, oW, oCh, decoded);
    ++iter;
  }
  std::printf("[loop] done after %zu generation%s\n", iter,
              iter == 1 ? "" : "s");
  return 0;
}

int main(int argc, char** argv) {
  std::string weights = "transformer";
  std::string prompt = "a red panda reading a newspaper";
  std::string text_weights, prompt_embeds_in, init_latents_in, dump_dir;
  std::string tokenizer_dir, text_encoder;
  std::string vae_weights, out = "qwen_image_21_out.png";
  std::string decode_latents_in, decode_latents_layout = "chw",
              decode_out_f32;
  int height = 1024, width = 1024;
  int steps = 8;
  bool steps_given = false;
  float guidance = 1.0f, flow_shift = 3.0f;
  std::string sigmas_mode = "driver";
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = true, vae_decode = true;
  bool prompt_given = false;

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
      prompt_given = true;
    } else if (a == "--tokenizer" && i + 1 < argc) {
      tokenizer_dir = argv[++i];
    } else if (a == "--text-encoder" && i + 1 < argc) {
      text_encoder = argv[++i];
    } else if (a == "--text-weights" && i + 1 < argc) {
      text_weights = argv[++i];
    } else if (a == "--prompt-embeds-in" && i + 1 < argc) {
      prompt_embeds_in = argv[++i];
    } else if (a == "--init-latents-in" && i + 1 < argc) {
      init_latents_in = argv[++i];
    } else if (a == "--dump-steps" && i + 1 < argc) {
      dump_dir = argv[++i];
    } else if (a == "--vae-weights" && i + 1 < argc) {
      vae_weights = argv[++i];
    } else if (a == "--vae-decode") {
      vae_decode = true;
    } else if (a == "--no-vae-decode" || a == "--no-vae_decode") {
      vae_decode = false;
    } else if (a == "--decode-latents-in" && i + 1 < argc) {
      decode_latents_in = argv[++i];
    } else if (a == "--decode-latents-layout" && i + 1 < argc) {
      decode_latents_layout = argv[++i];
      if (decode_latents_layout != "chw" && decode_latents_layout != "token") {
        std::fprintf(stderr,
                     "error: --decode-latents-layout must be 'chw' or "
                     "'token' (got '%s')\n",
                     decode_latents_layout.c_str());
        return 1;
      }
    } else if (a == "--decode-out-f32" && i + 1 < argc) {
      decode_out_f32 = argv[++i];
    } else if (a == "--height" && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    } else if (a == "--width" && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
      steps_given = true;
    } else if ((a == "--cfg" || a == "--guidance") && i + 1 < argc) {
      guidance = std::stof(argv[++i]);
    } else if ((a == "--shift" || a == "--flow-shift") && i + 1 < argc) {
      flow_shift = std::stof(argv[++i]);
    } else if (a == "--sigmas" && i + 1 < argc) {
      sigmas_mode = argv[++i];
      if (sigmas_mode != "driver" && sigmas_mode != "turbo") {
        std::fprintf(stderr, "error: --sigmas must be 'driver' or 'turbo' "
                             "(got '%s')\n",
                     sigmas_mode.c_str());
        return 1;
      }
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--staged") {
      staged = true;
    } else if (a == "--no-staged" || a == "--no-stage") {
      staged = false;
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
    std::printf("[backend dit=%s encoder=%s vae=%s]\n",
#ifdef QWEN21_DRIVER_HAVE_DIT
                "qwen-image-21-dit",
#else
                "placeholder",
#endif
#ifdef QWEN21_DRIVER_HAVE_ENCODER
                "qwen",
#else
                "file/null",
#endif
#ifdef QWEN21_DRIVER_HAVE_VAE
                "qwen-image-21-vae"
#else
                "preview"
#endif
    );
#ifdef QWEN21_DRIVER_HAVE_DIT
    // True-pipeline DiT header is present: use the landed geometry
    // (QwenImage21::Config: the default ctor is the full 32x4096/64ch/ctx4096
    // config; tiny_test() is the 2x32/4ch/ctx16 smoke config).
    KokkosDNN::QwenImage21::Config cfg =
        tiny ? KokkosDNN::QwenImage21::Config::tiny_test()
             : KokkosDNN::QwenImage21::Config();
    if (tiny) {
      // tiny_test() is the 2x32/4ch/ctx16 smoke config (now validate-clean).
    }
    cfg.validate();
#else
    QW::Config cfg = tiny ? QW::Config::tiny_test() : QW::Config::qwen_image_21();
    cfg.validate();
#endif
    const int steps_eff = (!steps_given && tiny) ? 2 : steps;
    // --height/--width are output pixels (default 1024); each rounds to the
    // nearest 16*patch multiple (see resolve_pixel_geometry). --tiny pins
    // 4x4 latents (64px) regardless.
    PixelGeometry geo;
    if (tiny) {
      geo.px_h = geo.px_w = 4 * kPixelScale;
      geo.lat_h = geo.lat_w = 4;
    } else {
      geo = resolve_pixel_geometry(height, width, be_patch(cfg));
      if (geo.px_h != height || geo.px_w != width)
        std::printf("[size] rounded %dx%d px -> %dx%d px (latent %dx%d, "
                    "stride %dpx)\n",
                    height, width, geo.px_h, geo.px_w, geo.lat_h, geo.lat_w,
                    kPixelScale * be_patch(cfg));
    }
    const int H = geo.lat_h, W = geo.lat_w;
    const int pxH = geo.px_h, pxW = geo.px_w;
    const size_t C = be_latent_channels(cfg);
    std::printf("[model %s: %s]\n", tiny ? "tiny_test" : "qwen-image-2.1",
                be_label(cfg));

    // ---- VAE-only parity path (--decode-latents-in) ----
    // Bypasses encode + DiT load + sampler: golden latents go straight
    // through the runtime affine + VAE decode. Needs the real VAE.
    if (!decode_latents_in.empty()) {
      if (vae_weights.empty())
        throw std::runtime_error(
            "--decode-latents-in needs --vae-weights (implies decode)");
      if (tiny)
        throw std::runtime_error(
            "--decode-latents-in needs the real VAE; drop --tiny");
      run_vae_only(decode_latents_in, decode_latents_layout == "token", C,
                   (size_t)H, (size_t)W, vae_weights, out, decode_out_f32);
      Kokkos::finalize();
      return 0;
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

    // Default interactive loop: no --prompt (and no file/parity handoff) on
    // a TTY means prompt -> filename -> generate, indefinitely. Weights are
    // preloaded once into host memory and copied host->device per iteration
    // (see run_interactive_loop). Anything script-like -- an explicit
    // --prompt, a caption/latent file, --dump-steps, or non-TTY stdin --
    // takes the legacy one-off path below, so golden/CI runs are unchanged.
    const bool loop_mode =
        !prompt_given && caption_src.empty() && init_latents_in.empty() &&
        dump_dir.empty() && stdin_is_tty();
    if (loop_mode) {
      LoopFixedOpts lo;
      lo.weights = weights;
      lo.tokenizer_dir = tokenizer_dir;
      lo.text_encoder = text_encoder;
      lo.H = height;
      lo.W = width;
      lo.steps_eff = steps_eff;
      lo.guidance = guidance;
      lo.flow_shift = flow_shift;
      lo.sigmas_mode = sigmas_mode;
      lo.seed = seed;
      lo.vae_weights = vae_weights;
      lo.vae_decode = vae_decode;
      lo.out_default = out;
      lo.staged = staged;
      lo.tiny = tiny;
      const int lrc = run_interactive_loop(cfg, lo);
      Kokkos::finalize();
      return lrc;
    }

    // ---- Phase 1: encode (no DiT weights are loaded yet) ----
    EncoderStage enc = EncoderStage::run(cfg, prompt, caption_src,
                                         tokenizer_dir, text_encoder, tiny);
    log_encode_memory(enc);

    // In-memory handoff: the embeds (~L*Cc floats per branch, MiB-scale) are
    // separate device allocations from the GiB-scale encoder weights, so they
    // stay resident on the device while the encoder weights are released and
    // the DiT is loaded. No disk round-trip and no second process.
    if (staged) {
      // When the real Qwen encoder lands in this TU, its EncoderModel weights
      // are released here; the EncoderStage embeds are independent Views and
      // survive. Until then this is a fence + accounting point only.
      Kokkos::fence();
      QW::print_phase_memory("encode-released", enc.device_bytes(), 0);
      std::printf("[staged] keeping %.1f MiB of embeds resident on device "
                  "across the DiT load (no disk round-trip)\n",
                  (double)enc.device_bytes() / (1024.0 * 1024.0));
    } else {
      std::printf("[note] single-phase run (--no-staged): encoder embeds stay\n"
                  "                 live across the DiT load (no fenced, logged\n"
                  "                 encoder-release point)\n");
    }

    // ---- Phase 2: DiT load ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    // True-pipeline DiT (QwenImage21DitModel owns its weights + scratch).
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> dit_model(
        cfg);
    {
      auto& w = dit_model.weights();
      if (tiny) {
        w.random_init(seed);
        std::printf("[tiny random weights] layers=%d hidden=%d latent=%dx%d\n",
                    cfg.num_layers, cfg.hidden_dim, H, W);
      } else {
        std::printf("[loading %s]\n", weights.c_str());
        try {
          const std::vector<std::string> shards =
              resolve_shard_list(weights, "--weights");
          if (shards.size() > 1)
            std::printf("[loading %zu shards]\n", shards.size());
          w.load_safetensors(shards);
        } catch (const std::exception& e) {
          throw std::runtime_error(std::string(e.what()) +
                                   " (need Qwen-Image-2.1 weights or --tiny)");
        }
        std::printf("[loaded] layers=%d hidden=%d\n", cfg.num_layers,
                    cfg.hidden_dim);
      }
      QW::print_phase_memory("sample", w.resident_bytes(),
                             dit_model.scratch_bytes());
    }
    std::printf("[budget] DiT transient scratch grows during sampling "
                "(see README fit table)\n");
#else
    QW::Weights<Exec, DScalar> w;
    if (tiny) {
      QW::random_init(cfg, w, seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      throw std::runtime_error(
          "weights file '" + weights +
          "' needs the true Qwen-Image-2.1 DiT "
          "(KokkosDNN_QwenImage21Dit.hpp is not in this build); pass --tiny "
          "for the random-weights smoke run. See README for the expected "
          "checkpoint layout.");
    }
    QW::print_phase_memory("sample", w.resident_bytes(), w.resident_bytes());
#endif
    std::printf("[budget] full-checkpoint projection: DiT %.1f GB + VAE %.3f "
                "GB + activations vs 24GB (see README fit table)\n",
                QW::kDitBudgetBytes / 1e9, QW::kVaeBudgetBytes / 1e9);

    if (!vae_weights.empty()) {
      std::error_code ec0;
      const bool vae_is_dir = std::filesystem::is_directory(vae_weights, ec0);
      if (!vae_is_dir && !file_exists(vae_weights))
        throw std::runtime_error("vae weights file '" + vae_weights +
                                 "' not found");
#ifdef QWEN21_DRIVER_HAVE_VAE
      std::printf("[vae %s] validated%s\n", vae_weights.c_str(),
                  vae_decode ? "; decode enabled (default)"
                             : "; preview (pass --vae-decode to decode)");
#else
      std::printf("[vae %s] validated, but KokkosDNN_QwenImage21Vae.hpp is "
                  "not in this build; writing a grayscale preview instead\n",
                  vae_weights.c_str());
#endif
    } else if (vae_decode) {
      std::printf("[vae] VAE decode needs --vae-weights (else grayscale "
                  "preview); disabling decode for this run\n");
      vae_decode = false;
    }

    std::vector<float> x_init;
    if (!init_latents_in.empty()) {
      // --init-latents-in: raw float32 LE initial latents in the driver's
      // native layout ([C,H,W] row-major, flattened C-major as
      // lat[(c*H+h)*W+w], the same order the sampler packs into
      // x_d(0,c,h,w)). Values are
      // PRE-denormalization latent noise (no mean/std applied); validated
      // against C*H*W, deep-copied to device by the sample loop, and used
      // in place of the mt19937 sample.
      x_init = load_init_latents_raw(init_latents_in, C, (size_t)H, (size_t)W);
      std::printf("[init-latents %s] %zux%dx%d floats (fp32 LE [C,H,W])\n",
                  init_latents_in.c_str(), C, H, W);
    } else {
      x_init = QW::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    }
    // Sigma schedule: 'driver' (default, current behavior) builds the
    // nearest-grid schedule from --steps/--shift; 'turbo' uses the
    // checkpoint sample_sigmas verbatim (mu-shifted already, so shift is
    // 1.0/no-shift in that mode) plus a terminal 0, i.e. 8 Euler steps.
    std::vector<float> turbo_sigmas;
    if (sigmas_mode == "turbo") {
#ifdef QWEN21_DRIVER_HAVE_OPS
      for (double s : cfg.sample_sigmas)
        turbo_sigmas.push_back((float)s);
      turbo_sigmas.push_back(0.0f);
      std::printf("[schedule turbo] checkpoint sample_sigmas verbatim "
                  "(mu-shifted already; --shift/--steps ignored except for "
                  "this note)\n");
#else
      throw std::runtime_error(
          "--sigmas turbo needs the true QwenImage21::Config "
          "(KokkosDNN_QwenImage21Config.hpp is not in this build)");
#endif
    }
    const int steps_run =
        (sigmas_mode == "turbo") ? (int)turbo_sigmas.size() - 1 : steps_eff;
    if (sigmas_mode == "turbo" && steps_given && steps != steps_run)
      std::printf("[schedule turbo] ignoring --steps %d (checkpoint schedule "
                  "has %d steps)\n",
                  steps, steps_run);
    std::printf("[sample] %dx%d px (latent %dx%d) steps=%d cfg=%.3g shift=%.3g "
                "seed=%llu\n",
                pxH, pxW, H, W, steps_run, (double)guidance, (double)flow_shift,
                (unsigned long long)seed);
    // ---- Parity dumps (--dump-steps DIR): inert unless the flag is set ----
    DumpWriter dump(dump_dir);
    std::vector<float> dump_sigmas;  // exact schedule, incl. terminal 0
    bool dump_use_cfg = false;
    if (dump.active()) {
      // The driver-mode schedule is rebuilt here because the nearest-grid
      // snap can drop repeated entries: the real step count is
      // sigmas.size()-1, not necessarily --steps.
      dump_sigmas = (sigmas_mode == "turbo")
                        ? turbo_sigmas
                        : build_sigmas_nearest_grid(steps_eff, flow_shift);
      // The sampler's starting latents, exactly as handed to it (no
      // additional scaling).
      dump.write("initial_latents", x_init);
    }
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    // Per-step parity dump (--dump-steps): writes stepNN_noise_pred.f32 /
    // stepNN_latents.f32 (+ the cond/uncond branches under CFG) from the
    // host copies the loop takes only when this hook is set.
    StepDumpHook dump_hook;
    if (dump.active()) {
      dump_hook.want_branches = (guidance != 1.0f);
      dump_hook.use_cfg_out = &dump_use_cfg;
      dump_hook.on_step_done =
          [&dump](int i, const std::vector<float>& x_step,
                  const std::vector<float>& v_step,
                  const std::vector<float>& v_cond,
                  const std::vector<float>& v_uncond) {
            char name[64];
            std::snprintf(name, sizeof(name), "step%02d_noise_pred", i);
            dump.write(name, v_step);
            if (!v_cond.empty()) {
              std::snprintf(name, sizeof(name), "step%02d_noise_pred_cond", i);
              dump.write(name, v_cond);
              std::snprintf(name, sizeof(name), "step%02d_noise_pred_uncond",
                            i);
              dump.write(name, v_uncond);
            }
            std::snprintf(name, sizeof(name), "step%02d_latents", i);
            dump.write(name, x_step);
          };
    }
    const auto t0 = std::chrono::steady_clock::now();
    // Sample straight from the device-resident embeds staged above: no host
    // reload, no disk re-read. The host vectors (enc.embeds/neg_embeds) stay
    // as the fallback for the --prompt-embeds-in file path, which also lands
    // staged on-device via EncoderStage::run.
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_run,
        guidance, flow_shift, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        },
        (sigmas_mode == "turbo") ? &turbo_sigmas : nullptr, dump_hook);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("[done] %.2fs total, %.2fs/step\n", secs,
                secs / (double)std::max(steps_run, 1));

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

    if (dump.active()) {
      DumpManifest dm;
      dm.prompt = prompt;
      dm.prompt_source = enc.source;
      dm.model_label = be_label(cfg);
      dm.sigmas_mode = sigmas_mode;
      dm.sigmas = dump_sigmas;
      dm.height = pxH;
      dm.width = pxW;
      dm.latent_h = H;
      dm.latent_w = W;
      dm.steps = (int)dump_sigmas.size() - 1;
      dm.latent_channels = C;
      dm.prompt_len = enc.device.extent(0);
      dm.prompt_dim = enc.device.extent(1);
      dm.seed = (unsigned long long)seed;
      dm.guidance = (double)guidance;
      dm.use_cfg = dump_use_cfg;
      dm.staged = staged;
      if (dm.steps != steps_run)
        std::printf("[dump] note: snapped schedule has %d steps (driver "
                    "reported %d)\n",
                    dm.steps, steps_run);
      write_dump_manifest(dump, dm);
    }

    // Captions are no longer needed: drop host + device copies before decode.
    enc.release();
    // ---- Phase 3: decode (the DiT is released first, then fenced) ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    dit_model.weights().release();
    dit_model.release_scratch();
#else
    w.release();
#endif
    Kokkos::fence();
    QW::print_phase_memory("dit-released", 0, 0);
    size_t oH = 0, oW = 0;
    // Only consumed by the PNG path (KOKKOSDNN_ENABLE_IMAGE); the PPM
    // fallback below ignores it.
    [[maybe_unused]] int oCh = 1;
    std::vector<unsigned char> pix;
    bool decoded = false;
    if (vae_decode) {
#ifdef QWEN21_DRIVER_HAVE_VAE
      try {
        decoded = try_vae_decode_rgba(lat, C, (size_t)H, (size_t)W,
                                      vae_weights, pix, oH, oW, oCh);
        std::printf("[decode] VAE RGBA %zux%zux%d ready for output\n", oH, oW,
                    oCh);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: real VAE decode failed (%s); writing a "
                     "grayscale preview instead\n",
                     e.what());
      }
#else
      std::printf("[vae] VAE decode requested (default) but "
                  "KokkosDNN_QwenImage21Vae.hpp is not in this build; writing "
                  "a grayscale preview instead (need the VAE worktree "
                  "header + --vae-weights)\n");
#endif
    }
    if (!decoded) {
      pix = render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
      QW::print_phase_memory("decode-preview", 0, 0);
    }
    write_output_image(out, pix, oH, oW, oCh, decoded);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
