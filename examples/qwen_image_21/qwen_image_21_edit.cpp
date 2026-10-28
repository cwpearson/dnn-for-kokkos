// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 one-shot reference-image editing driver.
//
// Wires the Qwen-Image-2.1-Turbo edit pipeline stage by stage, mirroring the
// diffusers QwenImage21Pipeline reference (text-to-image + image-conditioned
// __call__ with use_kv_cache=True):
//
//   load RGBA condition images (repeatable --image, up to 10, shared by the
//   reference QwenImage21Pipeline image=[...] list) -> derive geometry per
//   image (reference calculate_dimensions at --output-resolution, /32
//   rounding; the target size derives from the LAST image's aspect) ->
//   resized vision copies (float Lanczos) + bit-exact VAE copies
//   (PIL-fixed-point Lanczos replica of preprocess()) -> joint text+vision
//   encode (edit template below) -> the VAE-encoded condition latents
//   (--ref-latents-in per image in --image order, else in-process VAE encode
//   of each shared resized condition image, else zeros) are staged as packed
//   cond tokens
//   alongside the target noise -> FlowEuler sample with the DiT
//   image-condition path ([cond|target] + img_shapes/img_mask) -> VAE decode.
//
// ONE-SHOT ONLY: no interactive loop; one edit per invocation (unlike
// qwen_image_21_generate.cpp, which loops on a TTY without --prompt).
//
// Joint text+vision encode is wired in-process: --ckpt (canonical
// checkpoint root) + --image runs the raw-text BPE of the edit template
// (the slot's single image_pad placeholder expanded in place to Nv vision
// ids, processor convention), the Qwen3-VL vision tower over the packed
// pixel rows, and the text tower forward_joint (slot replacement, true
// M-RoPE, DeepStack), dropping the system block. --prompt-embeds-in (joint
// torch dump) and null zeros remain as fallbacks with logged reasons.
//   * VAE encode (image -> (mode-mean)/std latents) runs in-process via
//     QwenImage21VaeModel::encode from <ckpt>/vae:
//     --ref-latents-in carries torch/golden-encoded reference latents
//     ([C,H,W] chw or torch [H*W,C] token, validated against the condition
//     grid) as an override, else zeros. Vision-slot rows come from the
//     joint build (minus drop_idx) or --vis-rows-in; without either, the
//     run falls back to target-only sampling with a logged reason.
//
// Shared implementation (EncoderStage, sample loop, DiT adapters, VAE
// decode, image writer, parity dumps) comes from
// qwen_image_21_common.hpp, so the two binaries cannot drift apart.
//
//   qwen_image_21_edit [--image in1.png --image in2.png ...] --prompt "..."
//                       [--ckpt ckpt_dir]
//                       [--output-resolution 1024]
//                       [--height H --width W]  (override output; /32 snap)
//                       [--prompt-embeds-in pe.f32]
//                       [--ref-latents-in ref1.f32 [--ref-latents-in ref2.f32 ...]
//                        [--ref-latents-layout chw|token]]
//                       [--init-latents-in n.f32]
//                       [--steps 8] [--cfg 1.0] [--shift 3.0]
//                       [--sigmas driver|turbo] [--seed 42]
//                       [--no-vae-decode]
//                       [--out edit.png] [--staged] [--dump-steps DIR]
//                       [--tiny]
#include "qwen_image_21_edit_helpers.hpp"
int main(int argc, char** argv) {
  std::vector<std::string> image_paths;
  std::string prompt = "turn the sketch into a photorealistic photo";
  std::string text_weights, prompt_embeds_in;
  std::vector<std::string> ref_latents_ins;
  std::string ref_latents_layout = "chw";
  std::string vis_rows_in;
  std::string init_latents_in, dump_dir;
  std::string ckpt;  // canonical checkpoint root (transformer/,
                     // text_encoder/, processor/, vae/)
  std::string out = "qwen_image_21_edit.png";
#ifdef KOKKOSDNN_ENABLE_IMAGE
  int output_resolution = 1024;
#endif
  int height = 0, width = 0;  // 0 = derive from the condition aspect
  int steps = 8;
  bool steps_given = false;
  float guidance = 1.0f, flow_shift = 3.0f;
  std::string sigmas_mode = "driver";
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = true, vae_decode = true;

  // Like the reference image=[...] list: repeatable, order-significant.
  constexpr size_t kMaxImages = 10;

  // Shared helper for flags that append to a list (repeatable --image and
  // --ref-latents-in, both in --image order).
  const auto need_list = [&](std::vector<std::string>& dst, const std::string& a,
                             int& i) {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
      return false;
    }
    dst.push_back(argv[++i]);
    return true;
  };

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
    if (a == "--image") {
      if (!need_list(image_paths, a, i)) return 1;
    } else if (a == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
#ifdef KOKKOSDNN_ENABLE_IMAGE
    } else if (a == "--output-resolution" && i + 1 < argc) {
      output_resolution = std::stoi(argv[++i]);
#endif
    } else if (a == "--text-weights" && i + 1 < argc) {
      text_weights = argv[++i];
    } else if (a == "--prompt-embeds-in" && i + 1 < argc) {
      prompt_embeds_in = argv[++i];
    } else if (a == "--ref-latents-in" && i + 1 < argc) {
      if (!need_list(ref_latents_ins, a, i)) return 1;
    } else if (a == "--ref-latents-layout" && i + 1 < argc) {
      ref_latents_layout = argv[++i];
      if (ref_latents_layout != "chw" && ref_latents_layout != "token") {
        std::fprintf(stderr,
                     "error: --ref-latents-layout must be 'chw' or 'token' "
                     "(got '%s')\n",
                     ref_latents_layout.c_str());
        return 1;
      }
    } else if (a == "--vis-rows-in" && i + 1 < argc) {
      vis_rows_in = argv[++i];
    } else if (a == "--init-latents-in" && i + 1 < argc) {
      init_latents_in = argv[++i];
    } else if (a == "--dump-steps" && i + 1 < argc) {
      dump_dir = argv[++i];
    } else if (a == "--ckpt") {
      if (!need(ckpt)) return 1;
    } else if (a == "--vae-decode") {
      vae_decode = true;
    } else if (a == "--no-vae-decode" || a == "--no-vae_decode") {
      vae_decode = false;
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
        std::fprintf(stderr,
                     "error: --sigmas must be 'driver' or 'turbo' (got '%s')\n",
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
      edit_usage();
      return 0;
    } else {
      std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n", a.c_str());
    }
  }

  if (image_paths.empty() && !tiny) {
    std::fprintf(stderr, "error: --image PATH is required (omit only with --tiny,"
                         " which uses a synthesized test card)\n");
    return 1;
  }
  if (image_paths.size() > kMaxImages) {
    std::fprintf(stderr,
                 "error: got %zu --image flags (the reference accepts up to "
                 "%zu; pass one set of condition images per edit)\n",
                 image_paths.size(), kMaxImages);
    return 1;
  }
  if (!ref_latents_ins.empty() && !image_paths.empty() &&
      ref_latents_ins.size() != image_paths.size()) {
    // Single-image runs keep the old single-file form; multi-image runs
    // take one file per --image in the same order (or none for VAE encode).
    std::fprintf(stderr,
                 "error: got %zu --ref-latents-in files for %zu --image "
                 "flags (pass 0 or exactly %zu, one per --image in order)\n",
                 ref_latents_ins.size(), image_paths.size(), image_paths.size());
    return 1;
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
    KokkosDNN::QwenImage21::Config cfg =
        tiny ? KokkosDNN::QwenImage21::Config::tiny_test()
             : KokkosDNN::QwenImage21::Config();
    cfg.validate();
#else
    QW::Config cfg = tiny ? QW::Config::tiny_test() : QW::Config::qwen_image_21();
    cfg.validate();
#endif
    const int steps_eff = (!steps_given && tiny) ? 2 : steps;
    const size_t C = be_latent_channels(cfg);
    // Full runs need the canonical checkpoint; --tiny skips it.
    if (!tiny) require_ckpt_dir(ckpt);

    // ---- Condition images: load + reference geometry ----
    // One resize per image feeds both the vision copy and the VAE copy
    // (reference: input_images for the text encoder, vae_images for the
    // VAE). The pipeline reads RGBA (non-RGBA inputs are converted up);
    // the vision copy is flattened over white for the encoder while the
    // VAE keeps all four channels. Each input size comes from the reference
    // calculate_dimensions (output_resolution^2 area at that image's own
    // aspect, /32 rounding); explicit --height/--width override only the
    // OUTPUT size (floored to /32 like the reference). The default output
    // size derives from the LAST image's aspect, matching the reference.
    std::vector<int> cond_ws, cond_hs;  // resized condition pixels per image
    std::vector<int> in_lat_hs, in_lat_ws;
    std::vector<std::vector<float>> vis_pixels_list;  // packed vision rows [-1,1]
    std::vector<std::vector<float>> vae_chw_list;     // resized RGBA [-1,1] CHW
    {
#ifdef KOKKOSDNN_ENABLE_IMAGE
      namespace KI = KokkosDNN::Image;
      // --tiny with no --image exercises the smoke path with one
      // synthesized card; the cond DiT path stays off under --tiny.
      const bool synthetic = image_paths.empty();
      const size_t n_load = synthetic ? 1 : image_paths.size();
      if (!synthetic && output_resolution <= 0 && !tiny)
        throw std::runtime_error("--output-resolution must be > 0");
      cond_ws.reserve(n_load);
      cond_hs.reserve(n_load);
      in_lat_hs.reserve(n_load);
      in_lat_ws.reserve(n_load);
      vis_pixels_list.reserve(n_load);
      vae_chw_list.reserve(n_load);
      for (size_t ii = 0; ii < n_load; ++ii) {
        KI::HostImageF32 img("qwen21edit_img", 0, 0, 0);
        KI::HostImageU8 img_u8("qwen21edit_rgba_u8", 0, 0, 0);
        if (synthetic) {
          // Weight-free smoke card (tiny only): 64x64 RGBA gradient.
          img = KI::HostImageF32("qwen21edit_testcard", 64, 64, 4);
          for (size_t y = 0; y < 64; ++y)
            for (size_t x = 0; x < 64; ++x) {
              img(y, x, 0) = (float)x / 63.0f;
              img(y, x, 1) = (float)y / 63.0f;
              img(y, x, 2) = 0.5f;
              img(y, x, 3) = 1.0f;
            }
          std::printf("[image synthesized test card] 64x64 RGBA (--tiny, no "
                      "--image)\n");
        } else {
          img = KI::load_image_f32(image_paths[ii], /*desired_channels=*/4);
          img_u8 = KI::load_image_u8(image_paths[ii], /*desired_channels=*/4);
        }
        const size_t iw = img.extent(1), ih = img.extent(0);
        std::printf("[image %zu/%zu %s] %zux%zu RGBA\n", ii + 1, n_load,
                    synthetic ? "(synthetic)" : image_paths[ii].c_str(), iw, ih);
        int cond_w = 0, cond_h = 0;
        if (tiny) {
          cond_w = cond_h = 4 * kPixelScale;
        } else {
          edit_calculate_dimensions(output_resolution * output_resolution,
                                    (double)iw / (double)ih, cond_w, cond_h);
          if (cond_w < 32 || cond_h < 32)
            throw std::runtime_error(
                "edit geometry: derived condition size too small");
        }
        KI::HostImageF32 rgb = edit_flatten_white_rgb(img);
        if ((size_t)cond_w != iw || (size_t)cond_h != ih) {
          rgb = edit_bicubic_resize_hwc(rgb, cond_h, cond_w);
          std::printf("[image %zu] resized %zux%zu -> %dx%d (float lanczos "
                      "vision copy; PIL-exact uint8 VAE copy)\n",
                      ii + 1, iw, ih, cond_w, cond_h);
        }
        // VAE copy: bit-exact replica of the reference preprocess() resize
        // (PIL LANCZOS on the uint8 RGBA image, no white flatten, no RGB
        // drop) then u8/255 normalized to [-1,1] like pil_to_numpy +
        // normalize. A float-domain resize here leaves ~2 LSB of mismatch
        // that the VAE encoder amplifies past the gate, so the uint8-exact
        // path above is required.
        std::vector<float> vae_chw;
        if (!tiny) {
          KI::HostImageU8 vrgba = img_u8;
          if ((size_t)cond_w != iw || (size_t)cond_h != ih)
            vrgba = edit_pil_resize_rgba_u8(img_u8, cond_h, cond_w);
          vae_chw.resize(4 * (size_t)cond_h * (size_t)cond_w);
          size_t vk = 0;
          for (size_t c = 0; c < 4; ++c)
            for (int y = 0; y < cond_h; ++y)
              for (int x = 0; x < cond_w; ++x)
                vae_chw[vk++] = static_cast<float>(vrgba((size_t)y, (size_t)x,
                                                          c)) /
                                255.0f * 2.0f - 1.0f;
        }
        // Processor normalize: (x - 0.5) / 0.5 over RGB.
        for (size_t y = 0; y < rgb.extent(0); ++y)
          for (size_t x = 0; x < rgb.extent(1); ++x)
            for (size_t c = 0; c < 3; ++c)
              rgb(y, x, c) = rgb(y, x, c) * 2.0f - 1.0f;
        std::vector<float> vis_pixels = edit_pack_patches_blockmajor(rgb, 16, 2);
        std::printf("[vision %zu] %dx%d px -> %zu patches (16px, block-major, "
                    "temporal x2)\n",
                    ii + 1, cond_w, cond_h, vis_pixels.size() / 1536);
        const int in_lat_h = cond_h / kPixelScale;
        const int in_lat_w = cond_w / kPixelScale;
        std::printf("[condition %zu] %dx%d px -> VAE grid %dx%d (x%d)\n",
                    ii + 1, cond_w, cond_h, in_lat_h, in_lat_w, kPixelScale);
        cond_ws.push_back(cond_w);
        cond_hs.push_back(cond_h);
        in_lat_hs.push_back(in_lat_h);
        in_lat_ws.push_back(in_lat_w);
        vis_pixels_list.push_back(std::move(vis_pixels));
        vae_chw_list.push_back(std::move(vae_chw));
      }
#else
      (void)cond_ws;
      (void)cond_hs;
      (void)in_lat_hs;
      (void)in_lat_ws;
      (void)vis_pixels_list;
      (void)vae_chw_list;
      throw std::runtime_error(
          "--image needs the image component "
          "(configure with -DKokkosDNN_ENABLE_IMAGE=ON)");
#endif
    }
    const size_t n_imgs = cond_ws.size();
    if (n_imgs == 0)
      throw std::runtime_error("edit: no condition images loaded");

    // Target geometry: explicit H/W override the derived condition size
    // (from the LAST image, matching the reference); --tiny pins 64px.
    // Driver stride snap is a no-op on /32 sizes.
    PixelGeometry geo;
    if (tiny) {
      geo.px_h = geo.px_w = 4 * kPixelScale;
      geo.lat_h = geo.lat_w = 4;
    } else {
      const int req_h = (height > 0) ? height : cond_hs.back();
      const int req_w = (width > 0) ? width : cond_ws.back();
      geo = resolve_pixel_geometry(req_h, req_w, be_patch(cfg));
      if (geo.px_h != req_h || geo.px_w != req_w)
        std::printf("[size] rounded %dx%d px -> %dx%d px (latent %dx%d, "
                    "stride %dpx)\n",
                    req_w, req_h, geo.px_w, geo.px_h, geo.lat_w, geo.lat_h,
                    kPixelScale * be_patch(cfg));
    }
    const int H = geo.lat_h, W = geo.lat_w;
    const int pxH = geo.px_h, pxW = geo.px_w;
    std::printf("[model %s: %s]\n", tiny ? "tiny_test" : "qwen-image-2.1",
                be_label(cfg));

    // Caption source: --prompt-embeds-in is an alias of --text-weights and
    // wins when both are given. It is resolved before the joint attempt so
    // the fallback path below is already known.
    std::string caption_src = text_weights;
    if (!prompt_embeds_in.empty()) {
      if (!text_weights.empty() && prompt_embeds_in != text_weights)
        std::fprintf(stderr,
                     "warning: --prompt-embeds-in wins over --text-weights\n");
      caption_src = prompt_embeds_in;
    }

    // Joint processor inputs for the in-process edit encode (token ids with
    // the image_pad expansions, vision rows, M-RoPE positions, pixel rows
    // per image). Built here (host-side, needs the BPE tables); consumed
    // by EncoderStage::run_qwen_joint_multi below. Empty unless the build
    // succeeds.
    std::vector<int> joint_ids, joint_rows, joint_pos3;
    std::vector<std::vector<float>> joint_pixels_list;
    std::vector<int> joint_Hps, joint_Wps;
    size_t joint_drop = 0;
    bool have_joint_inputs = false;
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER) && defined(KOKKOSDNN_ENABLE_IMAGE)
    if (!tiny && !ckpt.empty() && !image_paths.empty()) {
      try {
        namespace QE = KokkosDNN::QwenImage21Encoder;
        require_ckpt_dir(ckpt);
        const std::string tok_json =
            ckpt_sub(ckpt, "processor") + "/tokenizer.json";
        if (!file_exists(tok_json))
          throw std::runtime_error("--ckpt '" + ckpt +
                                   "' has no processor/tokenizer.json (need "
                                   "the canonical Qwen-Image-2.1 "
                                   "checkpoint dir)");
        const RawQwenTables tabs = load_raw_qwen_tables(tok_json);
        // No 512-row T2I cap on the joint path (real edits carry thousands
        // of vision tokens); bound only against absurd sequences.
        constexpr size_t kJointMax = 32768;
        const size_t max_len = kJointMax;
        // System block (drop prefix): same fixed block the T2I template
        // uses (qwen21_t2i_head minus the user prefix), so drop_idx matches
        // the reference _drop_idx.
        const std::string kUserPrefix = "<|im_start|>user\n";
        const std::string full_head = QE::qwen21_t2i_head();
        if (full_head.size() < kUserPrefix.size() ||
            full_head.compare(full_head.size() - kUserPrefix.size(),
                              kUserPrefix.size(), kUserPrefix) != 0)
          throw std::runtime_error(
              "joint encode: T2I head lost its user-role prefix");
        const std::string sys_block =
            full_head.substr(0, full_head.size() - kUserPrefix.size());
        const std::string head_text = qwen21_edit_head((int)n_imgs);
        // Tail follows the last slot's own <|vision_end|>: the reference
        // prompt_template_ti2i closes each vision slot exactly once, so no
        // second close is prepended here.
        const std::string tail_text = prompt + qwen21_edit_tail();
        const std::vector<int> sys_ids =
            encode_qwen_raw(tabs, sys_block, max_len);
        const std::vector<int> head_ids =
            encode_qwen_raw(tabs, head_text, max_len);
        const std::vector<int> tail_ids =
            encode_qwen_raw(tabs, tail_text, max_len);
        if (sys_ids.empty() || head_ids.size() <= sys_ids.size() ||
            !std::equal(sys_ids.begin(), sys_ids.end(), head_ids.begin()))
          throw std::runtime_error(
              "joint encode: head lost its system-block prefix");
        const auto pad_it = tabs.vocab.find("<|image_pad|>");
        int pad_id = -1;
        if (pad_it != tabs.vocab.end()) {
          pad_id = pad_it->second;
        } else {
          // Added tokens live in specials, not vocab: longest-match split
          // emits their ids directly (see encode_qwen_raw).
          for (const auto& kv : tabs.specials)
            if (kv.first == "<|image_pad|>") pad_id = kv.second;
        }
        if (pad_id < 0)
          throw std::runtime_error(
              "joint encode: tokenizer.json lacks <|image_pad|>");
        // One patch grid per image (patch 16px); each slot's single
        // <|image_pad|> placeholder expands in place to that image's Nv
        // vision ids, mirroring the processor convention (one pad token in
        // the raw text expands to Nv image tokens, slots in --image order).
        joint_Hps.assign(n_imgs, 0);
        joint_Wps.assign(n_imgs, 0);
        std::vector<size_t> joint_Nvs(n_imgs, 0), joint_Nps(n_imgs, 0);
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          joint_Hps[ii] = cond_hs[ii] / 16;
          joint_Wps[ii] = cond_ws[ii] / 16;
          if (joint_Hps[ii] <= 0 || joint_Wps[ii] <= 0 ||
              joint_Hps[ii] % 2 != 0 || joint_Wps[ii] % 2 != 0)
            throw std::runtime_error(
                "joint encode: patch grid must tile the 2x2 merge");
          joint_Nvs[ii] =
              (size_t)(joint_Hps[ii] / 2) * (size_t)(joint_Wps[ii] / 2);
          joint_Nps[ii] = (size_t)joint_Hps[ii] * (size_t)joint_Wps[ii];
          if (vis_pixels_list[ii].size() != joint_Nps[ii] * 1536)
            throw std::runtime_error(
                "joint encode: pixel rows != patches*1536");
        }
        size_t pad_count = 0;
        for (int id : head_ids)
          if (id == pad_id) ++pad_count;
        if (pad_count != n_imgs)
          throw std::runtime_error(
              "joint encode: head must hold one <|image_pad|> placeholder "
              "per --image (got " +
              std::to_string(pad_count) + " pads for " +
              std::to_string(n_imgs) + " images)");
        joint_ids.reserve(head_ids.size() + tail_ids.size());
        size_t img_slot = 0;
        for (size_t k = 0; k < head_ids.size(); ++k) {
          if (head_ids[k] == pad_id) {
            for (size_t t = 0; t < joint_Nvs[img_slot]; ++t) {
              joint_rows.push_back((int)joint_ids.size());
              joint_ids.push_back(pad_id);
            }
            ++img_slot;
          } else {
            joint_ids.push_back(head_ids[k]);
          }
        }
        joint_ids.insert(joint_ids.end(), tail_ids.begin(), tail_ids.end());
        // No 512-row T2I cap here: real edits carry thousands of vision
        // tokens (2048px -> 4096 per image). Bound only against absurd
        // sequences.
        if (joint_ids.size() > 32768)
          throw std::runtime_error(
              "joint encode: sequence exceeds 32768 tokens");
        // M-RoPE over the fully expanded sequence: text runs advance, each
        // image's vision block holds T with H/W grids (inter-slot text such
        // as the space between slots advances normally).
        std::vector<std::pair<int, int>> joint_grids;
        joint_grids.reserve(n_imgs);
        for (size_t ii = 0; ii < n_imgs; ++ii)
          joint_grids.emplace_back(joint_Hps[ii] / 2, joint_Wps[ii] / 2);
        joint_pos3 = edit_rope_positions_multi(joint_ids.size(), joint_rows,
                                               joint_grids, joint_Nvs);
        joint_pixels_list = vis_pixels_list;
        joint_drop = sys_ids.size();
        have_joint_inputs = true;
        size_t joint_Nv_total = 0;
        for (size_t n : joint_Nvs) joint_Nv_total += n;
        std::printf("[edit template] %zu ids (%zu vision across %zu images), "
                    "drop_idx=%zu, tok_vocab=%zu\n",
                    joint_ids.size(), joint_Nv_total, n_imgs, joint_drop,
                    tabs.vocab_size);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: joint processor build failed (%s); falling "
                     "back to %s\n",
                     e.what(), caption_src.c_str());
      }
    }
#endif  // HAVE_ENCODER && HAVE_TOKENIZER && IMAGE (joint path)

    // ---- Reference latents (VAE-encoded condition images) ----
    // Precedence, per image: --ref-latents-in (torch/golden
    // (latents-mean)/std, driver chw or torch token layout, one file per
    // --image in the same order) -> in-process VAE encode of that image's
    // shared resized copy (needs <ckpt>/vae + the VAE header) -> zeros with
    // a logged reason. Either way they are CARRIED (validated, dumped,
    // reported), then staged below as packed cond tokens for the DiT
    // image-condition path ([cond|target] + img_shapes/img_mask); without
    // vision-slot rows the run falls back to target-only sampling (still
    // dumped + reported).
    std::vector<std::vector<float>> ref_lats(n_imgs);
    if (!ref_latents_ins.empty()) {
      if (tiny)
        throw std::runtime_error(
            "--ref-latents-in needs real geometry; drop --tiny");
      if (ref_latents_ins.size() != n_imgs)
        throw std::runtime_error(
            "--ref-latents-in needs one file per --image (in --image order)");
      for (size_t ii = 0; ii < n_imgs; ++ii) {
        const std::string& f = ref_latents_ins[ii];
        ref_lats[ii] = load_decode_latents(f, C, (size_t)in_lat_hs[ii],
                                           (size_t)in_lat_ws[ii],
                                           ref_latents_layout == "token");
        std::printf("[ref-latents file %zu %s] %zux%dx%d floats (%s layout)\n",
                    ii + 1, f.c_str(), C, in_lat_hs[ii], in_lat_ws[ii],
                    ref_latents_layout.c_str());
      }
    } else if (!tiny && !ckpt.empty()) {
      bool any_pixels = false;
      for (const auto& v : vae_chw_list) any_pixels = any_pixels || !v.empty();
      if (any_pixels) {
#ifdef QWEN21_DRIVER_HAVE_VAE
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          if (vae_chw_list[ii].empty()) {
            ref_lats[ii].assign(
                C * (size_t)in_lat_hs[ii] * (size_t)in_lat_ws[ii], 0.0f);
            std::printf("[ref-latents null %zu] no pixels: zeros %zux%dx%d\n",
                        ii + 1, C, in_lat_hs[ii], in_lat_ws[ii]);
            continue;
          }
          try {
            ref_lats[ii] = try_vae_encode_image(
                vae_chw_list[ii], (size_t)cond_hs[ii], (size_t)cond_ws[ii],
                ckpt);
            std::printf(
                "[ref-latents vae-encode %zu] %zux%dx%d floats (in-process "
                "RGBA [-1,1] -> (mode-mean)/std)\n",
                ii + 1, C, in_lat_hs[ii], in_lat_ws[ii]);
          } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "warning: in-process VAE encode failed for image %zu "
                         "(%s); using zeros\n",
                         ii + 1, e.what());
            ref_lats[ii].assign(
                C * (size_t)in_lat_hs[ii] * (size_t)in_lat_ws[ii], 0.0f);
            std::printf(
                "[ref-latents null %zu] VAE encode failed: zeros %zux%dx%d\n",
                ii + 1, C, in_lat_hs[ii], in_lat_ws[ii]);
          }
        }
#else
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          ref_lats[ii].assign(
              C * (size_t)in_lat_hs[ii] * (size_t)in_lat_ws[ii], 0.0f);
          std::printf("[ref-latents null %zu] VAE encode needs "
                      "KokkosDNN_QwenImage21Vae.hpp in this build: zeros "
                      "%zux%dx%d\n",
                      ii + 1, C, in_lat_hs[ii], in_lat_ws[ii]);
        }
#endif
      } else {
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          ref_lats[ii].assign(
              C * (size_t)in_lat_hs[ii] * (size_t)in_lat_ws[ii], 0.0f);
          const char* why = tiny ? "--tiny skips VAE encode (smoke model)"
                                 : "need --ckpt for VAE encode";
          std::printf("[ref-latents null %zu] %s: zeros %zux%dx%d\n", ii + 1,
                      why, C, in_lat_hs[ii], in_lat_ws[ii]);
        }
      }
    } else {
      for (size_t ii = 0; ii < n_imgs; ++ii) {
        ref_lats[ii].assign(
            C * (size_t)in_lat_hs[ii] * (size_t)in_lat_ws[ii], 0.0f);
        const char* why = tiny ? "--tiny skips VAE encode (smoke model)"
                               : "need --ckpt for VAE encode";
        std::printf("[ref-latents null %zu] %s: zeros %zux%dx%d\n", ii + 1,
                    why, C, in_lat_hs[ii], in_lat_ws[ii]);
      }
    }

    // Caption source (resolved above): --prompt-embeds-in wins over
    // --text-weights (joint text+vision embeddings for edit; L is inferred
    // from the file). Uncond stays zeros (the reference re-encodes the
    // negative prompt with the images under CFG; file/null parity stays
    // zeros like the generate driver).

    // ---- Phase 1: encode (no DiT weights are loaded yet) ----
    // Precedence: in-process joint Qwen forward (--ckpt + --image) ->
    // --prompt-embeds-in file -> null zeros. A throw inside the joint
    // attempt falls back to file/null with a logged reason instead of
    // failing.
    EncoderStage enc;
    bool joint_ok = false;  // true only when the joint embeds below are live
    if (have_joint_inputs) {
      try {
        EncoderStage::run_qwen_joint_multi(enc, cfg, joint_ids, joint_rows,
                                            joint_pos3, joint_pixels_list,
                                            joint_Hps, joint_Wps, joint_drop,
                                            ckpt);
        joint_ok = true;
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: in-process joint Qwen encode failed (%s); "
                     "falling back to %s\n",
                     e.what(), caption_src.empty() ? "null" : "file");
        enc = EncoderStage::run(cfg, prompt, caption_src, "", tiny,
                                     /*max_seq_override=*/32768);
      }
    } else {
      if (!tiny && !ckpt.empty() && !image_paths.empty())
        std::printf("[encode] joint Qwen forward needs the encoder + BPE "
                    "tables headers in this build; falling back to %s\n",
                    caption_src.empty() ? "null" : "file");
      else if (tiny && !ckpt.empty())
        std::printf("[encode] --tiny skips the joint Qwen forward "
                    "(random-weights smoke model); falling back to %s\n",
                    caption_src.empty() ? "null" : "file");
      enc = EncoderStage::run(cfg, prompt, caption_src, "", tiny,
                                     /*max_seq_override=*/32768);
    }
    log_encode_memory(enc);
    if (guidance != 1.0f)
      std::printf("[cfg] uncond stays zeros (the reference re-encodes the "
                  "negative prompt with images under CFG; file/null parity "
                  "stays zeros like the generate driver)\n");

    if (staged) {
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
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> dit_model(
        cfg);
    {
      auto& w = dit_model.weights();
      if (tiny) {
        w.random_init(seed);
        std::printf("[tiny random weights] layers=%d hidden=%d latent=%dx%d\n",
                    cfg.num_layers, cfg.hidden_dim, H, W);
      } else {
        const std::string dit_dir = ckpt_sub(ckpt, "transformer");
        std::printf("[loading %s]\n", dit_dir.c_str());
        try {
          const std::vector<std::string> shards =
              resolve_shard_list(dit_dir, "--ckpt");
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
          "checkpoint '" + ckpt_sub(ckpt, "transformer") +
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

    if (!tiny) {
      const std::string vae_dir = ckpt_sub(ckpt, "vae");
      std::error_code ec0;
      if (!std::filesystem::is_directory(vae_dir, ec0))
        throw std::runtime_error("--ckpt '" + ckpt +
                                 "' has no vae/ subdir (need the canonical "
                                 "Qwen-Image-2.1 checkpoint dir)");
#ifdef QWEN21_DRIVER_HAVE_VAE
      std::printf("[vae %s] validated%s\n", vae_dir.c_str(),
                  vae_decode ? "; decode enabled (default)"
                             : "; preview (pass --vae-decode to decode)");
#else
      std::printf("[vae %s] validated, but KokkosDNN_QwenImage21Vae.hpp is "
                  "not in this build; writing a grayscale preview instead\n",
                  vae_dir.c_str());
#endif
    } else if (vae_decode) {
      if (!ckpt.empty())
        std::printf("[vae] --tiny skips the VAE decode (random-weights smoke "
                    "model); writing a grayscale preview instead\n");
      else
        std::printf("[vae] VAE decode needs --ckpt (else grayscale "
                    "preview); disabling decode for this run\n");
      vae_decode = false;
    }

    std::vector<float> x_init;
    if (!init_latents_in.empty()) {
      x_init = load_init_latents_raw(init_latents_in, C, (size_t)H, (size_t)W);
      std::printf("[init-latents %s] %zux%dx%d floats (fp32 LE [C,H,W])\n",
                  init_latents_in.c_str(), C, H, W);
    } else {
      x_init = QW::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    }
    std::vector<float> turbo_sigmas;
    if (sigmas_mode == "turbo") {
#ifdef QWEN21_DRIVER_HAVE_OPS
      for (double s : cfg.sample_sigmas) turbo_sigmas.push_back((float)s);
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
    std::printf("[sample] edit %dx%d px (target latent %dx%d, %zu condition "
                "image%s) steps=%d cfg=%.3g shift=%.3g seed=%llu\n",
                pxH, pxW, H, W, n_imgs, n_imgs == 1 ? "" : "s", steps_run,
                (double)guidance, (double)flow_shift,
                (unsigned long long)seed);
    for (size_t ii = 0; ii < n_imgs; ++ii)
      std::printf("[sample] condition %zu/%zu: %dx%d px -> VAE grid %dx%d\n",
                  ii + 1, n_imgs, cond_ws[ii], cond_hs[ii], in_lat_hs[ii],
                  in_lat_ws[ii]);
    // ---- Edit image-condition state ([cond|target] DiT path) ----
    // Vision-slot rows in embed (post-drop) coordinates: the joint build
    // emits full-sequence positions, so subtract joint_drop; --vis-rows-in
    // is already in embed coordinates (range-checked at load). Staged cond
    // tokens pack each image's ref_lat host chw -> token rows (row t =
    // h*in_lat_w + w, the same order DitForward packs the target grid),
    // concatenated in --image order; img_shapes is
    // [[(1,hc0,wc0),(1,hc1,wc1),...,(1,H,W)]] (conds then target) and
    // img_mask is vision-slots True + text False, then N/4 target True.
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> cond_tok;
    std::vector<std::array<int, 3>> img_blocks;
    std::vector<char> img_mask;
    std::string img_shapes_str;
    size_t img_mask_true = 0;
    {
      std::vector<int> slot_rows;
      if (joint_ok && !joint_rows.empty()) {
        for (int r : joint_rows) slot_rows.push_back(r - (int)joint_drop);
      } else if (!vis_rows_in.empty() && !tiny) {
        slot_rows =
            load_vis_rows_raw(vis_rows_in, (size_t)enc.device.extent(0));
      }
      const size_t Senc = (size_t)enc.device.extent(0);
      std::vector<size_t> cond_Ncs(n_imgs, 0);
      size_t Nc_total = 0;
      for (size_t ii = 0; ii < n_imgs; ++ii) {
        cond_Ncs[ii] = (size_t)in_lat_hs[ii] * (size_t)in_lat_ws[ii];
        Nc_total += cond_Ncs[ii];
      }
      const size_t Nt = (size_t)H * (size_t)W;
      if (!tiny && !slot_rows.empty()) {
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          if (in_lat_hs[ii] <= 0 || in_lat_ws[ii] <= 0)
            throw std::runtime_error("edit cond: condition grid not derived");
        }
        // Each VLM slot stands for a 2x2 latent group: every cond latent
        // grid must tile the merge, and the target grid must hold N/4
        // slots (in_lat dims are /32-rounded, so even in practice).
        // The slot total covers all conditions (sum Nc_i/4): the joint
        // build emits Nv_i rows per image in --image order.
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          if (in_lat_hs[ii] % 2 != 0 || in_lat_ws[ii] % 2 != 0 || Nt % 4 != 0)
            throw std::runtime_error(
                "edit cond: latent grids must tile the 2x2 slots "
                "(each in_lat_h/w even, N % 4 == 0)");
        }
        if (slot_rows.size() * 4 != Nc_total)
          throw std::runtime_error(
              "edit cond: " + std::to_string(slot_rows.size()) +
              " vision slots != sum(Nc_i)/4 = " + std::to_string(Nc_total / 4));
        for (size_t k = 1; k < slot_rows.size(); ++k)
          if (slot_rows[k] <= slot_rows[k - 1])
            throw std::runtime_error(
                "edit cond: vision-slot rows must be strictly increasing");
        if (slot_rows.front() < 0 || slot_rows.back() >= (int)Senc)
          throw std::runtime_error(
              "edit cond: vision-slot row outside the staged S");
        cond_tok = Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec>(
            "qwen21edit_cond", 1, Nc_total, C);
        {
          auto hc = Kokkos::create_mirror_view(cond_tok);
          size_t t = 0;
          for (size_t ii = 0; ii < n_imgs; ++ii) {
            const int lhh = in_lat_hs[ii], lww = in_lat_ws[ii];
            const std::vector<float>& rl = ref_lats[ii];
            if (rl.size() != C * cond_Ncs[ii])
              throw std::runtime_error(
                  "edit cond: ref latents size != C*Hc*Wc");
            for (int hh = 0; hh < lhh; ++hh)
              for (int ww = 0; ww < lww; ++ww)
                for (size_t c = 0; c < C; ++c)
                  hc(0, t + (size_t)(hh * lww + ww), c) =
                      static_cast<DScalar>(rl[(c * (size_t)lhh +
                                              (size_t)hh) *
                                                 (size_t)lww +
                                             (size_t)ww]);
            t += cond_Ncs[ii];
          }
          Kokkos::deep_copy(cond_tok, hc);
        }
        Kokkos::fence();
        img_blocks.reserve(n_imgs + 1);
        for (size_t ii = 0; ii < n_imgs; ++ii)
          img_blocks.push_back({1, in_lat_hs[ii], in_lat_ws[ii]});
        img_blocks.push_back({1, H, W});
        img_mask.assign(Senc + Nt / 4, 0);
        for (int r : slot_rows) img_mask[(size_t)r] = 1;
        for (size_t v = Senc; v < img_mask.size(); ++v) img_mask[v] = 1;
        for (char c : img_mask) img_mask_true += c ? (size_t)1 : (size_t)0;
        std::ostringstream oss;
        oss << "[[";
        for (size_t ii = 0; ii < n_imgs; ++ii)
          oss << "(1," << in_lat_hs[ii] << "," << in_lat_ws[ii] << "),";
        oss << "(1," << H << "," << W << ")]]";
        img_shapes_str = oss.str();
        std::printf("[dit] cond path: %zu images Nc=%zu (", n_imgs, Nc_total);
        for (size_t ii = 0; ii < n_imgs; ++ii)
          std::printf("%s%dx%d", ii ? "+" : "", in_lat_hs[ii], in_lat_ws[ii]);
        std::printf(") N=%zu S=%zu T=%zu slots=%zu mask true=%zu/%zu "
                    "img_shapes %s\n",
                    Nt, Senc, Senc - slot_rows.size() + Nc_total + Nt,
                    slot_rows.size(), img_mask_true, img_mask.size(),
                    img_shapes_str.c_str());
      } else {
        std::printf("[dit] target-only sampling (no vision-slot rows%s)\n",
                    tiny ? " under --tiny"
                         : (joint_ok || !vis_rows_in.empty()
                                ? "; slot/geometry mismatch (see above)"
                                : "; need the joint build or --vis-rows-in "
                                  "for the cond path"));
      }
    }
    DumpWriter dump(dump_dir);
    std::vector<float> dump_sigmas;
    bool dump_use_cfg = false;
    if (dump.active()) {
      dump_sigmas = (sigmas_mode == "turbo")
                        ? turbo_sigmas
                        : build_sigmas_nearest_grid(steps_eff, flow_shift);
      dump.write("initial_latents", x_init);
      // Reference latents exactly as staged (zeros when neither
      // --ref-latents-in nor VAE encode supplied them): the packed tensors
      // the cond DiT path consumes. Single-image keeps the historical
      // "ref_latents" name; multi-image dumps one file per condition.
      if (n_imgs == 1) {
        dump.write("ref_latents", ref_lats[0]);
      } else {
        for (size_t ii = 0; ii < n_imgs; ++ii) {
          char name[64];
          std::snprintf(name, sizeof(name), "ref_latents_%zu", ii);
          dump.write(name, ref_lats[ii]);
        }
      }
      if (!img_mask.empty()) {
        std::vector<float> mf(img_mask.size());
        for (size_t i = 0; i < img_mask.size(); ++i)
          mf[i] = img_mask[i] ? 1.0f : 0.0f;
        dump.write("img_mask", mf);
      }
    }
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
    if (cond_tok.is_allocated()) {
      dit_forward.cond_tok = cond_tok;
      dit_forward.img_blocks = img_blocks;
      dit_forward.img_mask = img_mask;
    }
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
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
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_run,
        guidance, flow_shift, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        },
        (sigmas_mode == "turbo") ? &turbo_sigmas : nullptr, dump_hook,
        /*max_seq_override=*/32768);
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
      dm.prompt_source = std::string("edit:") + enc.source;
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
      dm.img_shapes = img_shapes_str;
      dm.img_mask_len = img_mask.size();
      dm.img_mask_true = img_mask_true;
      if (dm.steps != steps_run)
        std::printf("[dump] note: snapped schedule has %d steps (driver "
                    "reported %d)\n",
                    dm.steps, steps_run);
      write_dump_manifest(dump, dm);
    }

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
    [[maybe_unused]] int oCh = 1;
    std::vector<unsigned char> pix;
    bool decoded = false;
    if (vae_decode) {
#ifdef QWEN21_DRIVER_HAVE_VAE
      try {
        decoded = try_vae_decode_rgba(lat, C, (size_t)H, (size_t)W,
                                      ckpt, pix, oH, oW, oCh);
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
                  "header + --ckpt)\n");
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
