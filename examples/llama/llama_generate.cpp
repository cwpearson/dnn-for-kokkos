#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"
#include "KokkosDNN_Tokenizer.hpp"

// PoC driver: tiny random model by default; real weights via flags.
//   llama_generate [--tiny] [--weights a.safetensors ...]
//                  [--tokenizer tok.json] [--prompt "text"] [--max-new N]
//                  [--stop <id> ...] [--fp32] [--prefill-chunk N]
//                  [--config config.json] [--max-seq-len N]
//                  [--chat] [--system "text"] [--role user]
//                  [--add-bos | --no-add-bos] [--stop-string "str" ...]
//                  [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//                  [--repetition-penalty R] [--seed S]
//
// Default compute/storage is bfloat16 (Llama 3.1 8B-Instruct checkpoint dtype);
// --fp32 opts in. --prefill-chunk N caps prefill activations to
// O(N·dim) (0 = single-shot); default is Config::prefill_chunk_size.
//
// Default compute/storage is bfloat16 (Llama 3.1 8B-Instruct checkpoint dtype);
// --fp32 opts into FP32 for comparison. BF16 8B needs ~16 GiB (FP32
// ~32 GiB); RTX 3090 (24 GiB) fits BF16 but not FP32 — use --tiny or
// a big-GPU/CPU build for real checkpoints.
//
// --config overlays an HF config.json onto the selected base config
// (present keys win, missing keys keep hardcoded defaults); --max-seq-len
// caps the KV-cache context (validated against [1, 131072]).
//
// Chat: --chat wraps --prompt as one `role` (default "user") turn via the
// Llama 3.1 8B-Instruct template (BOS + header/content/<|eot_id|> blocks +
// open assistant header); --system prepends a system turn (implies chat).
// Raw mode (default) encodes --prompt directly.
//
// BOS: --add-bos (default) ensures exactly one leading <|begin_of_text|>
// (tokenizer bos id, else Config bos); --no-add-bos strips it. The chat
// path never double-counts the template's literal BOS.
//
// Stopping: by default the Instruct stop set {eos, eom, eot} from the
// tokenizer (or Config fallback without --tokenizer); --stop appends
// extra ids. --stop-string appends decoded-text stop strings (repeatable);
// generation is truncated before the earliest match.
template <class Scalar>
int run_model(KokkosDNN::Llama::Config cfg,
              const std::vector<std::string>& wfiles, bool tiny,
              const std::string& tok_path, const std::string& prompt,
              int max_new, const std::vector<int>& extra_stops,
              int prefill_chunk_override, bool chat_mode,
              const std::string& system_prompt, const std::string& role,
              bool want_bos, const std::vector<std::string>& stop_strings,
              KokkosDNN::SampleOptions samp_opt) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  if (tiny || wfiles.empty()) {
    KokkosDNN::Llama::random_init(cfg, w);
    std::printf("[tiny random weights] dim=%d layers=%d\n", cfg.dim,
                cfg.n_layers);
  } else {
    KokkosDNN::Llama::load_safetensors(cfg, wfiles, w);
    std::printf("[loaded %zu shard(s)]\n", wfiles.size());
  }

  std::vector<int> prompt_ids;
  std::vector<int> stops;
  const bool use_chat = chat_mode || !system_prompt.empty();
  if (!tok_path.empty()) {
    KokkosDNN::Llama::Tokenizer tok(tok_path);
    if (use_chat) {
      auto msgs = KokkosDNN::Llama::make_chat_messages(prompt, system_prompt,
                                                      role.empty() ? "user"
                                                                   : role);
      std::string rendered = KokkosDNN::Llama::render_chat_template(msgs, true);
      std::printf("[chat] %s", rendered.c_str());
      std::printf("\n");
      prompt_ids = KokkosDNN::Llama::encode_rendered_chat(tok, rendered,
                                                          want_bos);
    } else {
      prompt_ids =
          KokkosDNN::Llama::ensure_bos(tok.encode(prompt, want_bos), tok.bos(),
                                       want_bos);
    }
    std::printf("prompt tokens: %zu\n", prompt_ids.size());
    stops = {tok.eos(), tok.eom(), tok.eot()};
  } else {
    // byte-fallback prompt (no tokenizer file): deterministic ids.
    // Chat mode renders the template first so the byte stream matches;
    // BOS wiring mirrors the tokenizer path (exactly one or zero).
    std::string text = prompt;
    if (use_chat) {
      auto msgs = KokkosDNN::Llama::make_chat_messages(prompt, system_prompt,
                                                      role.empty() ? "user"
                                                                   : role);
      text = KokkosDNN::Llama::render_chat_template(msgs, true);
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
    stops = {cfg.eos_token, cfg.eom_token, cfg.eot_token};
  }
  stops.insert(stops.end(), extra_stops.begin(), extra_stops.end());
  std::printf("stop ids:");
  for (auto id : stops) std::printf(" %d", id);
  std::printf("\n");
  if (prefill_chunk_override >= 0) cfg.prefill_chunk_size = prefill_chunk_override;
  std::printf("prefill chunk: %d\n", cfg.prefill_chunk_size);
  if (prompt_ids.size() > (size_t)cfg.max_seq_len) {
    std::fprintf(stderr,
                 "error: prompt is %zu tokens but max_seq_len is %d "
                 "(chat template + byte-fallback without --tokenizer is "
                 "verbose; raise --max-seq-len or pass --tokenizer)\n",
                 prompt_ids.size(), cfg.max_seq_len);
    return 1;
  }

  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
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
    KokkosDNN::Llama::Tokenizer stok(tok_path);
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
    KokkosDNN::Llama::Tokenizer tok(tok_path);
    std::printf("decoded: %s\n", tok.decode(out).c_str());
    if (!stop_strings.empty()) {
      // Stop-strings are checked on decoded generated text (prompt
      // excluded so prompt content cannot false-trigger): truncate the
      // generation before the earliest match.
      std::vector<int> gen_ids(out.begin() + (long)prompt_ids.size(),
                               out.end());
      std::string gen_text = tok.decode(gen_ids);
      std::string cut =
          KokkosDNN::Llama::truncate_at_stop_strings(gen_text, stop_strings);
      if (cut.size() != gen_text.size()) {
        std::string hit;
        KokkosDNN::Llama::find_stop_string(gen_text, stop_strings, &hit);
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
    int max_new = 16;
    bool tiny = true;
    bool fp32 = false;
    std::vector<int> extra_stops;
    int prefill_chunk_override = -1;
    int max_seq_len_override = -1;
    bool chat_mode = false;
    std::string system_prompt;
    std::string role = "user";
    bool want_bos = true;  // --add-bos default (matches tokenizer TemplateProcessing)
    std::vector<std::string> stop_strings;
    KokkosDNN::SampleOptions samp_opt;  // greedy defaults (temp=0)

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
    KokkosDNN::Llama::Config cfg = KokkosDNN::Llama::Config::llama_3_1_8b();
    if (tiny) {
      cfg = KokkosDNN::Llama::Config::tiny_test();
      cfg.max_seq_len = 64;
    }
    if (!config_path.empty()) {
      try {
        cfg = KokkosDNN::Llama::Config::from_json_file(config_path, cfg);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --config '%s': %s\n",
                     config_path.c_str(), e.what());
        Kokkos::finalize();
        return 1;
      }
      std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d "
                  "vocab=%d eps=%g theta=%g max_seq=%d tie=%d\n",
                  config_path.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads,
                  cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size,
                  (double)cfg.rms_norm_eps, (double)cfg.rope_theta,
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
      rc = run_model<KokkosDNN::Llama::DefaultScalar>(cfg, wfiles, tiny,
                                                      tok_path, prompt, max_new,
                                                      extra_stops,
                                                      prefill_chunk_override,
                                                      chat_mode, system_prompt,
                                                      role, want_bos,
                                                      stop_strings, samp_opt);
    }
    Kokkos::finalize();
    return rc;
  }
}
