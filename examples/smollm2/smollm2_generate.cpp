// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include <Kokkos_Core.hpp>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_CliOptions.hpp"
#include "KokkosDNN_SmolLM2Config.hpp"
#include "KokkosDNN_SmolLM2Model.hpp"
#include "KokkosDNN_SmolLM2Tokenizer.hpp"
#include "KokkosDNN_SmolLM2Weights.hpp"

// PoC driver: tiny random model by default; real weights via flags.
//   smollm2_generate [--tiny] [--model smollm2-135m|smollm2-135m-instruct]
//                  [--weights a.safetensors ...]
//                  [--tokenizer tok.json] [--prompt "text"] [--max-new N]
//                  [--stop <id> ...] [--fp32] [--prefill-chunk N]
//                  [--config config.json] [--max-seq-len N]
//                  [--kv-page-size N] [--kv-int8]
//                  [--chat] [--system "text"] [--role user]
//                  [--add-bos | --no-add-bos] [--stop-string "str" ...]
//                  [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//                  [--repetition-penalty R] [--seed S]
//
// --model selects the base Config: smollm2-135m (default; aliases smollm2,
// 135m) or smollm2-135m-instruct (alias smollm2-instruct; case-insensitive,
// optional huggingfacetb- prefix, e.g. huggingfacetb-smollm2-135m).
// Llama model names are NOT accepted here (use the llama example driver).
// Passing --model explicitly opts out of the tiny default, so
// `--model smollm2-135m` alone runs the full-size base with random weights
// (no downloads); --tiny still forces tiny_test dims. --config overlays on
// top of the selected base (see below).
//
// Sampling defaults to temperature=0.6, top_p=0.9 (the core
// SampleOptions::llama_3_1_8b_instruct() preset: same temp/top-p the SmolLM2
// instruct card recommends); --temperature 0 restores greedy decode. Every
// sampling flag overrides the preset; seed is random unless --seed is
// given.
//
// Default compute/storage is bfloat16 (SmolLM2-135M checkpoint dtype);
// --fp32 opts into FP32 for comparison. BF16 135M needs ~0.27 GiB, so the
// full preset runs anywhere. --prefill-chunk N caps prefill activations to
// O(N·dim) (0 = single-shot); default is Config::prefill_chunk_size.
//
// --config overlays an HF config.json onto the selected base config
// (present keys win, missing keys keep hardcoded defaults); --max-seq-len
// caps the KV-cache context (validated against [1, 8192], the arch max).
// --kv-page-size N selects paged KV with block size N (0 = dense);
// --kv-int8 enables int8 KV (per-token per-head scales), --kv-fp restores
// FP. Paged dense is bit-exact vs dense; int8 trades small drift for ~2x
// fewer KV bytes (paged pools additionally grow on demand).
//
// Chat: --chat wraps --prompt as one `role` (default "user") turn via the
// SmolLM2-Instruct ChatML template (<|im_start|>/<|im_end|> blocks + open
// assistant header, with the "You are a helpful AI assistant."
// default-system injection when no system turn is present); --system
// prepends a system turn (implies chat). Raw mode (default) encodes
// --prompt directly.
//
// BOS: --add-bos (default) ensures exactly one leading BOS id (tokenizer
// bos id, else Config bos: <|endoftext|>/0 for Base, <|im_start|>/1 for
// Instruct); --no-add-bos strips it. The chat path never double-counts
// the template's literal BOS.
//
// Stopping: by default the single GPT-2 EOS id from the tokenizer
// (base {0=<|endoftext|>}, instruct {2=<|im_end|>}; Config fallback
// without --tokenizer). --stop appends extra ids. --stop-string appends
// decoded-text stop strings (repeatable); generation is truncated before
// the earliest match.
template <class Scalar>
int run_model(KokkosDNN::SmolLM2::Config cfg,
              const std::vector<std::string>& wfiles, bool tiny,
              const std::string& tok_path, const std::string& prompt,
              int max_new, const std::vector<int>& extra_stops,
              int prefill_chunk_override, bool chat_mode,
              const std::string& system_prompt, const std::string& role,
              bool want_bos, const std::vector<std::string>& stop_strings,
              KokkosDNN::SampleOptions samp_opt, int kv_page_override,
              bool kv_int8_override, bool kv_int8_touched) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::SmolLM2::Weights<Exec, Scalar> w;
  if (tiny || wfiles.empty()) {
    KokkosDNN::SmolLM2::random_init(cfg, w);
    std::printf("[%srandom weights] dim=%d layers=%d\n", tiny ? "tiny " : "",
                cfg.dim, cfg.n_layers);
  } else {
    KokkosDNN::SmolLM2::load_safetensors(cfg, wfiles, w);
    std::printf("[loaded %zu shard(s)]\n", wfiles.size());
  }

  std::vector<int> prompt_ids;
  std::vector<int> stops;
  const bool use_chat = chat_mode || !system_prompt.empty();
  if (!tok_path.empty()) {
    KokkosDNN::SmolLM2::Tokenizer tok(tok_path);
    if (use_chat) {
      auto msgs = KokkosDNN::SmolLM2::make_chat_messages(prompt, system_prompt,
                                                        role.empty() ? "user"
                                                                     : role);
      // SmolLM2-Instruct ChatML (default-system injection inside the
      // renderer); encode_chatml keeps the exactly-one-BOS invariant
      // with the GPT-2 BOS id.
      std::string rendered =
          KokkosDNN::SmolLM2::render_chatml_template(msgs, true);
      std::printf("[chatml] %s", rendered.c_str());
      std::printf("\n");
      prompt_ids =
          KokkosDNN::SmolLM2::encode_chatml(tok, msgs, true, want_bos);
    } else {
      prompt_ids =
          KokkosDNN::SmolLM2::ensure_bos(tok.encode(prompt, want_bos),
                                        tok.bos(), want_bos);
    }
    std::printf("prompt tokens: %zu\n", prompt_ids.size());
    // Single GPT-2 EOS stop (base 0, instruct 2 via tokenizer /
    // sibling tokenizer_config.json).
    stops = {tok.eos()};
  } else {
    // byte-fallback prompt (no tokenizer file): deterministic ids.
    // Chat mode renders the ChatML template first so the byte stream
    // matches; BOS wiring mirrors the tokenizer path (exactly one or zero).
    std::string text = prompt;
    if (use_chat) {
      auto msgs = KokkosDNN::SmolLM2::make_chat_messages(prompt, system_prompt,
                                                        role.empty() ? "user"
                                                                     : role);
      text = KokkosDNN::SmolLM2::render_chatml_template(msgs, true);
      std::printf("[chatml, no tokenizer] %s\n", text.c_str());
    }
    if (want_bos) prompt_ids.push_back(cfg.bos_token % cfg.vocab_size);
    for (unsigned char c : text) prompt_ids.push_back(c % cfg.vocab_size);
    std::printf("[no tokenizer] byte-fallback prompt tokens: %zu\n",
                prompt_ids.size());
    stops = {cfg.eos_token};
  }
  stops.insert(stops.end(), extra_stops.begin(), extra_stops.end());
  std::printf("stop ids:");
  for (auto id : stops) std::printf(" %d", id);
  std::printf("\n");
  if (prefill_chunk_override >= 0) cfg.prefill_chunk_size = prefill_chunk_override;
  std::printf("prefill chunk: %d\n", cfg.prefill_chunk_size);
  if (kv_page_override >= 0) cfg.kv_page_size = kv_page_override;
  if (kv_int8_touched) cfg.kv_int8 = kv_int8_override;
  try {
    cfg.validate();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: invalid KV-cache config: %s\n", e.what());
    return 1;
  }
  {
    const double mib =
        (double)cfg.kv_cache_bytes(sizeof(Scalar)) / (1024.0 * 1024.0);
    std::printf("kv cache: %s (page %d), %s, full-window %.1f MiB\n",
                cfg.kv_page_size > 0 ? "paged" : "dense", cfg.kv_page_size,
                cfg.kv_int8 ? "int8" : "fp", mib);
  }
  if (prompt_ids.size() > (size_t)cfg.max_seq_len) {
    std::fprintf(stderr,
                 "error: prompt is %zu tokens but max_seq_len is %d "
                 "(chat template + byte-fallback without --tokenizer is "
                 "verbose; raise --max-seq-len or pass --tokenizer)\n",
                 prompt_ids.size(), cfg.max_seq_len);
    return 1;
  }

  KokkosDNN::SmolLM2::Model<Exec, Scalar> model(cfg, w);
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
    KokkosDNN::SmolLM2::Tokenizer stok(tok_path);
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
    KokkosDNN::SmolLM2::Tokenizer tok(tok_path);
    std::printf("decoded: %s\n", tok.decode(out).c_str());
    if (!stop_strings.empty()) {
      // Stop-strings are checked on decoded generated text (prompt
      // excluded so prompt content cannot false-trigger): truncate the
      // generation before the earliest match.
      std::vector<int> gen_ids(out.begin() + (long)prompt_ids.size(),
                               out.end());
      std::string gen_text = tok.decode(gen_ids);
      std::string cut =
          KokkosDNN::SmolLM2::truncate_at_stop_strings(gen_text, stop_strings);
      if (cut.size() != gen_text.size()) {
        std::string hit;
        KokkosDNN::SmolLM2::find_stop_string(gen_text, stop_strings, &hit);
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
    KokkosDNN::Common::Cli::TextGenOptions o;
    o.samp = KokkosDNN::SampleOptions::llama_3_1_8b_instruct();
    std::string model_name = "smollm2-135m";  // --model base (default base)
    bool model_touched = false;  // explicit --model opts out of tiny default
    int kv_page_override = -1;
    bool kv_int8_override = false, kv_int8_touched = false;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (KokkosDNN::Common::Cli::try_parse_shared(a, argv, i, argc, o)) {
        continue;
      } else if (a == "--model" && i + 1 < argc) {
        std::string m = argv[++i];
        for (auto& ch : m) ch = (char)std::tolower((unsigned char)ch);
        if (m.rfind("huggingfacetb-", 0) == 0)
          m = m.substr(14);  // optional HF org prefix
        if (m == "smollm2-135m" || m == "smollm2" || m == "135m") {
          model_name = "smollm2-135m";
        } else if (m == "smollm2-135m-instruct" || m == "smollm2-instruct") {
          model_name = "smollm2-135m-instruct";
        } else {
          std::fprintf(stderr,
                       "error: --model '%s': expected "
                       "smollm2-135m|smollm2-135m-instruct "
                       "(aliases smollm2, 135m, smollm2-instruct, "
                       "case-insensitive, optional huggingfacetb- prefix); "
                       "Llama models are not accepted by this driver "
                       "(use the llama example driver)\n",
                       argv[i]);
          Kokkos::finalize();
          return 1;
        }
        model_touched = true;
      } else if (a == "--kv-page-size" && i + 1 < argc) {
        kv_page_override = std::stoi(argv[++i]);
      } else if (a == "--kv-int8") {
        kv_int8_override = true;
        kv_int8_touched = true;
      } else if (a == "--kv-fp") {
        kv_int8_override = false;
        kv_int8_touched = true;
      }
    }

    KokkosDNN::Common::Cli::resolve_random_seed(o);

    KokkosDNN::print_configuration(std::cout);
    // Explicit --model opts out of the tiny default so
    // `--model smollm2-135m` alone runs the full-size base with random
    // weights; explicit --tiny still forces tiny_test dims below.
    if (model_touched && !o.tiny_explicit) o.tiny = false;
    KokkosDNN::SmolLM2::Config cfg;
    if (model_name == "smollm2-135m-instruct")
      cfg = KokkosDNN::SmolLM2::Config::smollm2_135m_instruct();
    else
      cfg = KokkosDNN::SmolLM2::Config::smollm2_135m();
    std::printf("[model %s]\n", model_name.c_str());
    if (o.tiny) {
      cfg = KokkosDNN::SmolLM2::Config::tiny_test();
      cfg.max_seq_len = 64;
    }
    if (!o.config_path.empty()) {
      try {
        cfg = KokkosDNN::SmolLM2::Config::from_json_file(o.config_path, cfg);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --config '%s': %s\n",
                     o.config_path.c_str(), e.what());
        Kokkos::finalize();
        return 1;
      }
      std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d "
                  "vocab=%d eps=%g theta=%g max_seq=%d tie=%d\n",
                  o.config_path.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads,
                  cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size,
                  (double)cfg.rms_norm_eps, (double)cfg.rope_theta,
                  cfg.max_seq_len, (int)cfg.tie_word_embeddings);
    }
    if (o.max_seq_len > 0) cfg.max_seq_len = o.max_seq_len;
    try {
      cfg.validate();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: invalid model config: %s\n", e.what());
      Kokkos::finalize();
      return 1;
    }
    int rc;
    if (o.fp32) {
      std::printf("[dtype fp32]\n");
      rc = run_model<float>(cfg, o.weights, o.tiny, o.tokenizer_path, o.prompt,
                            o.max_new, o.extra_stops, o.prefill_chunk,
                            o.chat_mode, o.system_prompt, o.role, o.want_bos,
                            o.stop_strings, o.samp, kv_page_override,
                            kv_int8_override, kv_int8_touched);
    } else {
      std::printf("[dtype bfloat16]\n");
      rc = run_model<KokkosDNN::SmolLM2::DefaultScalar>(
          cfg, o.weights, o.tiny, o.tokenizer_path, o.prompt, o.max_new,
          o.extra_stops, o.prefill_chunk, o.chat_mode, o.system_prompt,
          o.role, o.want_bos, o.stop_strings, o.samp, kv_page_override,
          kv_int8_override, kv_int8_touched);
    }
    Kokkos::finalize();
    return rc;
  }
}
