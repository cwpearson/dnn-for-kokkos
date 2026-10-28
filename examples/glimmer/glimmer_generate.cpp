// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include <Kokkos_Core.hpp>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_GlimmerConfig.hpp"
#include "KokkosDNN_GlimmerModel.hpp"
#include "KokkosDNN_GlimmerTokenizer.hpp"
#include "KokkosDNN_GlimmerWeights.hpp"

// POSIX glob for --weights patterns (multi-file sharded checkpoints).
#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#include <glob.h>
#define KOKKOSDNN_GLIMMER_HAS_GLOB 1
#else
#define KOKKOSDNN_GLIMMER_HAS_GLOB 0
#endif

namespace {
// Expand shell-style --weights patterns (e.g. "model-*.safetensors").
// Patterns matching nothing are kept literally so the loader reports the
// missing file by name. Results are sorted and deduplicated.
std::vector<std::string> expand_weight_patterns(
    const std::vector<std::string>& pats) {
  std::vector<std::string> out;
#if KOKKOSDNN_GLIMMER_HAS_GLOB
  for (auto& p : pats) {
    glob_t g;
    int rc = ::glob(p.c_str(), 0, nullptr, &g);
    if (rc != 0 || g.gl_pathc == 0) {
      out.push_back(p);
    } else {
      for (size_t i = 0; i < g.gl_pathc; ++i) out.emplace_back(g.gl_pathv[i]);
    }
    ::globfree(&g);
  }
#else
  out = pats;
#endif
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// A single .gguf file (case-insensitive suffix) selects the GGUF loading
// path (llama.cpp tensor naming + dequantization); anything else is a
// safetensors shard (possibly a glob pattern, expanded above).
bool is_gguf_path(const std::string& p) {
  if (p.size() < 5) return false;
  std::string ext = p.substr(p.size() - 5);
  for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
  return ext == ".gguf";
}
}  // namespace

// PoC driver: tiny random model by default; real weights via flags.
//   glimmer_generate [--tiny] [--model glimmer-30b]
//                    [--weights model.safetensors (globs accepted) |
//                               model.gguf (single file, llama.cpp naming)]
//                    [--tokenizer tokenizer.json] [--prompt "text"]
//                    [--max-new N] [--stop <id> ...] [--fp32]
//                    [--prefill-chunk N] [--config config.json]
//                    [--max-seq-len N] [--list-tensors]
//                    [--chat] [--system "text"] [--role user]
//                    [--add-bos | --no-add-bos] [--stop-string "str" ...]
//                    [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//                    [--repetition-penalty R] [--seed S]
//
// --model selects the base Config (only glimmer-30b; aliases glimmer, 30b,
// muse-glimmer-30b, glimmer-30b-heretic; case-insensitive, optional muse- /
// darkc0de- prefix). Passing --model explicitly opts out of the tiny
// default, so `--model glimmer-30b` alone runs the full-size model with
// random weights (no downloads); --tiny still forces tiny_test dims.
// --config overlays on top of the selected base (the full HF config.json
// with text_config, or a bare text_config object; see below).
//
// Sampling defaults to greedy (the upstream generation_config.json sets
// do_sample=false, no temperature/top_k/top_p); any sampling flag opts into
// that sampler. Seed is random unless --seed is given.
//
// Default compute/storage is bfloat16 (heretic checkpoint dtype);
// --fp32 opts into FP32 for comparison. BF16 30B needs ~60 GiB, so the
// full preset needs a large-GPU or CPU host allocation. --prefill-chunk N
// caps prefill activations to O(N·dim) (0 = single-shot); default is
// Config::prefill_chunk_size.
//
// --config overlays an HF config.json onto the selected base config
// (present keys win, missing keys keep hardcoded defaults); --max-seq-len
// caps the KV-cache context (validated against [1, 131072], the arch max).
// --list-tensors prints the tensor names stored in --weights files
// (useful to inspect partial/sharded downloads) and exits. A single
// --weights entry ending in .gguf (case-insensitive) instead loads a
// llama.cpp GGUF checkpoint (dequantized on the host; F32/F16/BF16 exact,
// Q8_0/Q4_K/Q6_K supported, anything else throws); mixing .gguf with
// safetensors shards is an error. See README.md for the GGUF
// conversion-trap handling (norm shift, Q/K re-permute, synthesized QK
// norms, metadata validation).
//
// BOS: --add-bos (default) ensures exactly one leading BOS id
// (<|begin_of_text|>/200000); --no-add-bos strips it.
//
// Chat: --chat wraps --prompt as one `role` (default "user") turn via the
// Muse-style template (<|begin_of_text|> + <|start|>{role}<|message|>
// {content}<|eot|> blocks + open assistant header); --system prepends a
// system turn (implies chat); --messages is a minimal alias wrapping its
// value as the user turn (implies chat). NOTE: upstream ships no
// chat_template in tokenizer_config.json, so this framing is an ASSUMPTION
// inferred from the tokenizer_config.json `response_template` (documented
// in KokkosDNN_GlimmerTokenizer.hpp); raw mode (default) encodes --prompt
// directly.
//
// Stopping: by default the generation_config stop set {eos 200001,
// eot 200008} (resolved through the tokenizer where possible);
// --stop appends extra ids. --stop-string appends decoded-text stop strings
// (repeatable); generation is truncated before the earliest match.
template <class Scalar>
int run_model(KokkosDNN::Glimmer::Config cfg,
              const std::vector<std::string>& wfiles, bool tiny,
              const std::string& tok_path, const std::string& prompt,
              int max_new, const std::vector<int>& extra_stops,
              int prefill_chunk_override, bool chat_mode,
              const std::string& system_prompt, const std::string& role,
              bool want_bos, const std::vector<std::string>& stop_strings,
              KokkosDNN::SampleOptions samp_opt) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::Glimmer::Weights<Exec, Scalar> w;
  if (tiny || wfiles.empty()) {
    KokkosDNN::Glimmer::random_init(cfg, w);
    std::printf("[%srandom weights] dim=%d layers=%d\n", tiny ? "tiny " : "",
                cfg.dim, cfg.n_layers);
  } else if (wfiles.size() == 1 && is_gguf_path(wfiles[0])) {
    KokkosDNN::Glimmer::load_gguf(cfg, wfiles[0], w);
    std::printf("[loaded %s]\n", wfiles[0].c_str());
  } else {
    for (auto& fp : wfiles)
      if (is_gguf_path(fp))
        throw std::runtime_error(
            "GGUF loading takes a single .gguf file: got '" + fp +
            "' among " + std::to_string(wfiles.size()) +
            " --weights entries (mixing .gguf with safetensors shards is "
            "unsupported)");
    KokkosDNN::Glimmer::load_safetensors(cfg, wfiles, w);
    std::printf("[loaded %zu shard(s)]\n", wfiles.size());
  }

  std::vector<int> prompt_ids;
  std::vector<int> stops;
  const bool use_chat = chat_mode || !system_prompt.empty();
  if (!tok_path.empty()) {
    KokkosDNN::Glimmer::Tokenizer tok(tok_path);
    if (use_chat) {
      auto msgs = KokkosDNN::Glimmer::make_chat_messages(
          prompt, system_prompt, role.empty() ? "user" : role);
      std::string rendered =
          KokkosDNN::Glimmer::render_chat_template(msgs, true);
      std::printf("[chat] %s\n", rendered.c_str());
      prompt_ids =
          KokkosDNN::Glimmer::encode_rendered_chat(tok, rendered, want_bos);
    } else {
      prompt_ids = KokkosDNN::Glimmer::ensure_bos(tok.encode(prompt, want_bos),
                                                 tok.bos(), want_bos);
    }
    std::printf("prompt tokens: %zu\n", prompt_ids.size());
    // generation_config.json eos_token_id [200001, 200008], resolved
    // through the tokenizer where possible.
    stops = {tok.eos(), tok.eot()};
  } else {
    // byte-fallback prompt (no tokenizer file): deterministic ids.
    // Chat mode renders the template first so the byte stream matches;
    // BOS wiring mirrors the tokenizer path (exactly one or zero).
    std::string text = prompt;
    if (use_chat) {
      auto msgs = KokkosDNN::Glimmer::make_chat_messages(
          prompt, system_prompt, role.empty() ? "user" : role);
      text = KokkosDNN::Glimmer::render_chat_template(msgs, true);
      std::printf("[chat, no tokenizer] %s\n", text.c_str());
    }
    if (want_bos) prompt_ids.push_back(cfg.bos_token % cfg.vocab_size);
    for (unsigned char c : text) prompt_ids.push_back(c % cfg.vocab_size);
    // The rendered chat text starts with the literal "<|begin_of_text|>"
    // bytes here (no special-token split without a tokenizer), so collapse
    // an accidental double-BOS only in the single-BOS edge case where the
    // byte stream is empty: nothing further to do.
    std::printf("[no tokenizer] byte-fallback prompt tokens: %zu\n",
                prompt_ids.size());
    stops = {cfg.eos_token, 200008};  // generation_config eos ids
  }
  stops.insert(stops.end(), extra_stops.begin(), extra_stops.end());
  std::printf("stop ids:");
  for (auto id : stops) std::printf(" %d", id);
  std::printf("\n");
  if (prefill_chunk_override >= 0) cfg.prefill_chunk_size = prefill_chunk_override;
  std::printf("prefill chunk: %d\n", cfg.prefill_chunk_size);
  try {
    cfg.validate();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: invalid model config: %s\n", e.what());
    return 1;
  }
  {
    const double mib =
        (double)cfg.kv_cache_bytes(sizeof(Scalar)) / (1024.0 * 1024.0);
    std::printf("kv cache: dense, full-window %.1f MiB\n", mib);
  }
  if (prompt_ids.size() > (size_t)cfg.max_seq_len) {
    std::fprintf(stderr,
                 "error: prompt is %zu tokens but max_seq_len is %d "
                 "(raise --max-seq-len or pass --tokenizer)\n",
                 prompt_ids.size(), cfg.max_seq_len);
    return 1;
  }

  KokkosDNN::Glimmer::Model<Exec, Scalar> model(cfg, w);
  std::printf("sampling: temp=%g top_k=%d top_p=%g min_p=%g rep_pen=%g seed=%llu%s\n",
              (double)samp_opt.temperature, samp_opt.top_k,
              (double)samp_opt.top_p, (double)samp_opt.min_p,
              (double)samp_opt.repetition_penalty,
              (unsigned long long)samp_opt.seed,
              samp_opt.is_greedy() ? " [greedy]" : "");
  KokkosDNN::HostRNG rng(samp_opt.seed);
  std::vector<int> out;
  if (!tok_path.empty()) {
    // Tokenizer for streaming decode: constructed once, reused per token.
    KokkosDNN::Glimmer::Tokenizer stok(tok_path);
    std::printf("decoded (streaming): ");
    std::fflush(stdout);
    auto on_token = [&](int id) {
      std::string piece = stok.decode(std::vector<int>{id});
      std::fputs(piece.c_str(), stdout);
      std::fflush(stdout);
    };
    out = model.generate(prompt_ids, max_new, stops, samp_opt, &rng, on_token);
    std::printf("\n");
    std::fflush(stdout);
  } else {
    std::printf("generated (streaming):");
    std::fflush(stdout);
    auto on_token = [&](int id) {
      std::printf(" %d", id);
      std::fflush(stdout);
    };
    out = model.generate(prompt_ids, max_new, stops, samp_opt, &rng, on_token);
    std::printf("\n");
  }
  std::printf("generated %zu ids:", out.size());
  for (auto id : out) std::printf(" %d", id);
  std::printf("\n");
  if (!tok_path.empty()) {
    KokkosDNN::Glimmer::Tokenizer tok(tok_path);
    std::printf("decoded: %s\n", tok.decode(out).c_str());
    if (!stop_strings.empty()) {
      // Stop-strings are checked on decoded generated text (prompt
      // excluded so prompt content cannot false-trigger): truncate the
      // generation before the earliest match.
      std::vector<int> gen_ids(out.begin() + (long)prompt_ids.size(),
                               out.end());
      std::string gen_text = tok.decode(gen_ids);
      std::string cut =
          KokkosDNN::Glimmer::truncate_at_stop_strings(gen_text, stop_strings);
      if (cut.size() != gen_text.size()) {
        std::string hit;
        KokkosDNN::Glimmer::find_stop_string(gen_text, stop_strings, &hit);
        std::printf("stop-string hit: '%s'\n", hit.c_str());
      }
      std::printf("decoded (stop-strings applied): %s\n", cut.c_str());
    }
  } else if (!stop_strings.empty()) {
    std::fprintf(stderr,
                 "warning: --stop-string needs --tokenizer (decoded-text "
                 "check); ignoring %zu stop-string(s)\n",
                 stop_strings.size());
  }
  Kokkos::fence();
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::vector<std::string> wfiles, wpatterns;
    std::string tok_path, prompt = "Hello, I am";
    std::string config_path;
    std::string model_name = "glimmer-30b";  // --model base (only model)
    bool model_touched = false;  // explicit --model opts out of tiny default
    bool tiny_explicit = false;  // explicit --tiny still forces tiny_test
    int max_new = 16;
    bool tiny = true;
    bool fp32 = false;
    bool list_tensors = false;
    std::vector<int> extra_stops;
    int prefill_chunk_override = -1;
    int max_seq_len_override = -1;
    bool chat_mode = false;
    std::string system_prompt;
    std::string role = "user";
    bool want_bos = true;  // --add-bos default (matches TemplateProcessing)
    std::vector<std::string> stop_strings;
    // Upstream generation_config.json: do_sample=false (greedy), no
    // temperature/top_k/top_p. Sampling flags below opt into samplers.
    KokkosDNN::SampleOptions samp_opt;
    bool seed_given = false;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--weights" && i + 1 < argc) {
        // comma/space separated list in one arg, or repeated flags; entries
        // may be glob patterns (expanded after parsing).
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          wpatterns.push_back(argv[++i]);
        tiny = false;
      } else if (a == "--tokenizer" && i + 1 < argc) {
        tok_path = argv[++i];
      } else if (a == "--prompt" && i + 1 < argc) {
        prompt = argv[++i];
      } else if (a == "--max-new" && i + 1 < argc) {
        max_new = std::stoi(argv[++i]);
      } else if (a == "--stop" && i + 1 < argc) {
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          extra_stops.push_back(std::stoi(argv[++i]));
      } else if (a == "--tiny") {
        tiny = true;
        tiny_explicit = true;
      } else if (a == "--model" && i + 1 < argc) {
        std::string m = argv[++i];
        for (auto& ch : m) ch = (char)std::tolower((unsigned char)ch);
        for (const char* pre : {"darkc0de-", "muse-"}) {
          if (m.rfind(pre, 0) == 0) {
            m = m.substr(std::string(pre).size());
            break;
          }
        }
        if (m == "glimmer-30b" || m == "glimmer" || m == "30b" ||
            m == "glimmer-30b-heretic") {
          model_name = "glimmer-30b";
        } else {
          std::fprintf(stderr,
                       "error: --model '%s': expected "
                       "glimmer-30b "
                       "(aliases glimmer, 30b, muse-glimmer-30b, "
                       "glimmer-30b-heretic, case-insensitive, optional "
                       "muse-/darkc0de- prefix)\n",
                       argv[i]);
          Kokkos::finalize();
          return 1;
        }
        model_touched = true;
      } else if (a == "--chat") {
        chat_mode = true;
      } else if (a == "--system" && i + 1 < argc) {
        system_prompt = argv[++i];
      } else if (a == "--role" && i + 1 < argc) {
        role = argv[++i];
      } else if (a == "--messages" && i + 1 < argc) {
        // Minimal alias: wrap the value as the user turn (implies chat).
        prompt = argv[++i];
        chat_mode = true;
      } else if (a == "--add-bos") {
        want_bos = true;
      } else if (a == "--no-add-bos") {
        want_bos = false;
      } else if (a == "--stop-string" && i + 1 < argc) {
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          stop_strings.push_back(argv[++i]);
      } else if (a == "--prefill-chunk" && i + 1 < argc) {
        prefill_chunk_override = std::stoi(argv[++i]);
      } else if (a == "--list-tensors") {
        list_tensors = true;
      } else if (a == "--config" && i + 1 < argc) {
        config_path = argv[++i];
      } else if (a == "--max-seq-len" && i + 1 < argc) {
        max_seq_len_override = std::stoi(argv[++i]);
      } else if (a == "--fp32") {
        fp32 = true;
      } else if (a == "--temperature" && i + 1 < argc) {
        samp_opt.temperature = std::stof(argv[++i]);
      } else if (a == "--top-k" && i + 1 < argc) {
        samp_opt.top_k = std::stoi(argv[++i]);
      } else if (a == "--top-p" && i + 1 < argc) {
        samp_opt.top_p = std::stof(argv[++i]);
      } else if (a == "--min-p" && i + 1 < argc) {
        samp_opt.min_p = std::stof(argv[++i]);
      } else if (a == "--repetition-penalty" && i + 1 < argc) {
        samp_opt.repetition_penalty = std::stof(argv[++i]);
      } else if (a == "--seed" && i + 1 < argc) {
        samp_opt.seed = (uint64_t)std::stoull(argv[++i]);
        seed_given = true;
      }
    }

    if (!seed_given) {
      std::random_device rd;
      samp_opt.seed = (uint64_t(rd()) << 32) | uint64_t(rd());
    }

    KokkosDNN::print_configuration(std::cout);
    wfiles = expand_weight_patterns(wpatterns);
    // --list-tensors is a header-only inspection (no model build).
    if (list_tensors) {
      if (wfiles.empty()) {
        std::fprintf(stderr, "error: --list-tensors needs --weights <files>\n");
        Kokkos::finalize();
        return 1;
      }
      try {
        std::vector<std::string> names;
        if (wfiles.size() == 1 && is_gguf_path(wfiles[0]))
          names = KokkosDNN::Glimmer::list_gguf_tensors(wfiles[0]);
        else
          names = KokkosDNN::Glimmer::list_checkpoint_tensors(wfiles);
        std::printf("%zu tensor(s):\n", names.size());
        for (auto& n : names) std::printf("  %s\n", n.c_str());
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --list-tensors: %s\n", e.what());
        Kokkos::finalize();
        return 1;
      }
      Kokkos::finalize();
      return 0;
    }
    // Explicit --model opts out of the tiny default so
    // `--model glimmer-30b` alone runs the full-size model with random
    // weights; explicit --tiny still forces tiny_test dims below.
    if (model_touched && !tiny_explicit) tiny = false;
    KokkosDNN::Glimmer::Config cfg = KokkosDNN::Glimmer::Config::glimmer_30b();
    std::printf("[model %s]\n", model_name.c_str());
    if (tiny) {
      cfg = KokkosDNN::Glimmer::Config::tiny_test();
      cfg.max_seq_len = 64;
    }
    if (!config_path.empty()) {
      try {
        cfg = KokkosDNN::Glimmer::Config::from_json_file(config_path, cfg);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --config '%s': %s\n",
                     config_path.c_str(), e.what());
        Kokkos::finalize();
        return 1;
      }
      std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d "
                  "vocab=%d hdim=%d eps=%g post_eps=%g s_theta=%g sw=%d "
                  "max_seq=%d tie=%d\n",
                  config_path.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads,
                  cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size, cfg.head_dim,
                  (double)cfg.rms_norm_eps, (double)cfg.post_norm_eps,
                  (double)cfg.rope_theta, cfg.sliding_window,
                  cfg.max_seq_len, (int)cfg.tie_word_embeddings);
    }
    if (max_seq_len_override > 0) cfg.max_seq_len = max_seq_len_override;
    try {
      cfg.validate();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: invalid model config: %s\n", e.what());
      Kokkos::finalize();
      return 1;
    }
    int rc;
    if (fp32) {
      std::printf("[dtype fp32]\n");
      rc = run_model<float>(cfg, wfiles, tiny, tok_path, prompt, max_new,
                            extra_stops, prefill_chunk_override, chat_mode,
                            system_prompt, role, want_bos, stop_strings,
                            samp_opt);
    } else {
      std::printf("[dtype bfloat16]\n");
      rc = run_model<KokkosDNN::Glimmer::DefaultScalar>(
          cfg, wfiles, tiny, tok_path, prompt, max_new, extra_stops,
          prefill_chunk_override, chat_mode, system_prompt, role, want_bos,
          stop_strings, samp_opt);
    }
    Kokkos::finalize();
    return rc;
  }
}
