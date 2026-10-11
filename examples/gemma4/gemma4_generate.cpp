#include <Kokkos_Core.hpp>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_Gemma4Config.hpp"
#include "KokkosDNN_Gemma4Model.hpp"
#include "KokkosDNN_Gemma4Tokenizer.hpp"
#include "KokkosDNN_Gemma4Weights.hpp"

// PoC driver: tiny random model by default; real weights via flags.
//   gemma4_generate [--tiny] [--model gemma4-e2b-it]
//                   [--weights model.safetensors]
//                   [--tokenizer tokenizer.json] [--prompt "text"]
//                   [--max-new N] [--stop <id> ...] [--fp32]
//                   [--prefill-chunk N] [--config config.json]
//                   [--max-seq-len N] [--list-tensors]
//                   [--add-bos | --no-add-bos] [--stop-string "str" ...]
//                   [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//                   [--repetition-penalty R] [--seed S]
//
// --model selects the base Config (only gemma4-e2b-it; aliases gemma4,
// e2b-it; case-insensitive, optional google- prefix). Llama/SmolLM2 model
// names are NOT accepted here (use those example drivers). Passing --model
// explicitly opts out of the tiny default, so `--model gemma4-e2b-it`
// alone runs the full-size model with random weights (no downloads);
// --tiny still forces tiny_test dims. --config overlays on top of the
// selected base (the full HF config.json with text_config, or a bare
// text_config object; see below).
//
// Sampling defaults to the upstream generation_config.json
// (temperature=1.0, top_k=64, top_p=0.95); --temperature 0 restores greedy
// decode. Every sampling flag overrides the preset; seed defaults to 0 so
// runs repeat unless --seed is given.
//
// Default compute/storage is bfloat16 (E2B-it checkpoint dtype);
// --fp32 opts into FP32 for comparison. BF16 E2B-it needs ~10 GiB, so the
// full preset needs a large-GPU or CPU host allocation. --prefill-chunk N
// caps prefill activations to O(N·dim) (0 = single-shot); default is
// Config::prefill_chunk_size.
//
// --config overlays an HF config.json onto the selected base config
// (present keys win, missing keys keep hardcoded defaults); --max-seq-len
// caps the KV-cache context (validated against [1, 131072], the arch max).
// --list-tensors prints the tensor names stored in --weights files
// (useful to inspect partial/sharded downloads) and exits.
//
// BOS: --add-bos (default) ensures exactly one leading BOS id (<bos>/2);
// --no-add-bos strips it.
//
// Stopping: by default the generation_config set {1 (<eos>), 106, 50}
// (from Config::default_stop_ids). --stop appends extra ids.
// --stop-string appends decoded-text stop strings (repeatable);
// generation is truncated before the earliest match.
//
// NOTE: no --chat flag: the Gemma4 canonical chat template
// (chat_template.jinja) is a tool-calling/channel template out of scope
// for this text-only step; raw prompts are encoded directly.
template <class Scalar>
int run_model(KokkosDNN::Gemma4::Config cfg,
              const std::vector<std::string>& wfiles, bool tiny,
              const std::string& tok_path, const std::string& prompt,
              int max_new, const std::vector<int>& extra_stops,
              int prefill_chunk_override, bool want_bos,
              const std::vector<std::string>& stop_strings,
              KokkosDNN::SampleOptions samp_opt) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::Gemma4::Weights<Exec, Scalar> w;
  if (tiny || wfiles.empty()) {
    KokkosDNN::Gemma4::random_init(cfg, w);
    std::printf("[%srandom weights] dim=%d layers=%d\n", tiny ? "tiny " : "",
                cfg.dim, cfg.n_layers);
  } else {
    KokkosDNN::Gemma4::load_safetensors(cfg, wfiles, w);
    std::printf("[loaded %zu shard(s)]\n", wfiles.size());
  }

  std::vector<int> prompt_ids;
  std::vector<int> stops;
  if (!tok_path.empty()) {
    KokkosDNN::Gemma4::Tokenizer tok(tok_path);
    prompt_ids =
        KokkosDNN::Gemma4::ensure_bos(tok.encode(prompt, want_bos),
                                      tok.bos(), want_bos);
    std::printf("prompt tokens: %zu\n", prompt_ids.size());
    // generation_config.json eos_token_ids [1, 106, 50], resolved through
    // the tokenizer where possible (all three are plain ids here).
    stops = cfg.default_stop_ids;
  } else {
    // byte-fallback prompt (no tokenizer file): deterministic ids.
    if (want_bos) prompt_ids.push_back(cfg.bos_token % cfg.vocab_size);
    for (unsigned char c : prompt) prompt_ids.push_back(c % cfg.vocab_size);
    std::printf("[no tokenizer] byte-fallback prompt tokens: %zu\n",
                prompt_ids.size());
    stops = cfg.default_stop_ids;
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
    std::printf("kv cache: dense, %d sliding + %d full caches, "
                "full-window %.1f MiB\n",
                cfg.num_sliding_caches(), cfg.num_full_caches(), mib);
  }
  if (prompt_ids.size() > (size_t)cfg.max_seq_len) {
    std::fprintf(stderr,
                 "error: prompt is %zu tokens but max_seq_len is %d "
                 "(raise --max-seq-len or pass --tokenizer)\n",
                 prompt_ids.size(), cfg.max_seq_len);
    return 1;
  }

  KokkosDNN::Gemma4::Model<Exec, Scalar> model(cfg, w);
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
    KokkosDNN::Gemma4::Tokenizer stok(tok_path);
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
    KokkosDNN::Gemma4::Tokenizer tok(tok_path);
    std::printf("decoded: %s\n", tok.decode(out).c_str());
    if (!stop_strings.empty()) {
      // Stop-strings are checked on decoded generated text (prompt
      // excluded so prompt content cannot false-trigger): truncate the
      // generation before the earliest match.
      std::vector<int> gen_ids(out.begin() + (long)prompt_ids.size(),
                               out.end());
      std::string gen_text = tok.decode(gen_ids);
      std::string cut =
          KokkosDNN::Gemma4::truncate_at_stop_strings(gen_text, stop_strings);
      if (cut.size() != gen_text.size()) {
        std::string hit;
        KokkosDNN::Gemma4::find_stop_string(gen_text, stop_strings, &hit);
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
    std::vector<std::string> wfiles;
    std::string tok_path, prompt = "Hello, I am";
    std::string config_path;
    std::string model_name = "gemma4-e2b-it";  // --model base (only model)
    bool model_touched = false;  // explicit --model opts out of tiny default
    bool tiny_explicit = false;  // explicit --tiny still forces tiny_test
    int max_new = 16;
    bool tiny = true;
    bool fp32 = false;
    bool list_tensors = false;
    std::vector<int> extra_stops;
    int prefill_chunk_override = -1;
    int max_seq_len_override = -1;
    bool want_bos = true;  // --add-bos default (matches HF readings)
    std::vector<std::string> stop_strings;
    // Upstream generation_config.json: do_sample, temperature=1.0,
    // top_k=64, top_p=0.95.
    KokkosDNN::SampleOptions samp_opt;
    samp_opt.temperature = 1.0f;
    samp_opt.top_k = 64;
    samp_opt.top_p = 0.95f;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--weights" && i + 1 < argc) {
        // comma/space separated list in one arg, or repeated flags
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          wfiles.push_back(argv[++i]);
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
        if (m.rfind("google-", 0) == 0)
          m = m.substr(7);  // optional HF org prefix
        if (m == "gemma4-e2b-it" || m == "gemma4" || m == "e2b-it" ||
            m == "gemma-4-e2b-it") {
          model_name = "gemma4-e2b-it";
        } else {
          std::fprintf(stderr,
                       "error: --model '%s': expected "
                       "gemma4-e2b-it "
                       "(aliases gemma4, e2b-it, gemma-4-e2b-it, "
                       "case-insensitive, optional google- prefix); "
                       "Llama/SmolLM2 models are not accepted by this driver "
                       "(use those example drivers)\n",
                       argv[i]);
          Kokkos::finalize();
          return 1;
        }
        model_touched = true;
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
      }
    }

    KokkosDNN::print_configuration(std::cout);
    // --list-tensors is a header-only inspection (no model build).
    if (list_tensors) {
      if (wfiles.empty()) {
        std::fprintf(stderr, "error: --list-tensors needs --weights <files>\n");
        Kokkos::finalize();
        return 1;
      }
      try {
        auto names = KokkosDNN::Gemma4::list_checkpoint_tensors(wfiles);
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
    // `--model gemma4-e2b-it` alone runs the full-size model with random
    // weights; explicit --tiny still forces tiny_test dims below.
    if (model_touched && !tiny_explicit) tiny = false;
    KokkosDNN::Gemma4::Config cfg =
        KokkosDNN::Gemma4::Config::gemma4_e2b_it();
    std::printf("[model %s]\n", model_name.c_str());
    if (tiny) {
      cfg = KokkosDNN::Gemma4::Config::tiny_test();
      cfg.max_seq_len = 64;
    }
    if (!config_path.empty()) {
      try {
        cfg = KokkosDNN::Gemma4::Config::from_json_file(config_path, cfg);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --config '%s': %s\n",
                     config_path.c_str(), e.what());
        Kokkos::finalize();
        return 1;
      }
      std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d "
                  "vocab=%d eps=%g s_theta=%g f_theta=%g max_seq=%d tie=%d\n",
                  config_path.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads,
                  cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size,
                  (double)cfg.rms_norm_eps, (double)cfg.rope_theta_sliding,
                  (double)cfg.rope_theta_full, cfg.max_seq_len,
                  (int)cfg.tie_word_embeddings);
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
                            extra_stops, prefill_chunk_override, want_bos,
                            stop_strings, samp_opt);
    } else {
      std::printf("[dtype bfloat16]\n");
      rc = run_model<KokkosDNN::Gemma4::DefaultScalar>(
          cfg, wfiles, tiny, tok_path, prompt, max_new, extra_stops,
          prefill_chunk_override, want_bos, stop_strings, samp_opt);
    }
    Kokkos::finalize();
    return rc;
  }
}
