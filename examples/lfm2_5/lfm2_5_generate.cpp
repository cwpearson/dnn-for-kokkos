#include <Kokkos_Core.hpp>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Model.hpp"
#include "KokkosDNN_LFM25Tokenizer.hpp"
#include "KokkosDNN_LFM25Weights.hpp"

// PoC driver: tiny random model by default; real weights via --weights.
//   lfm2_5_generate [--tiny] [--weights model.gguf]
//                   [--tokenizer tok.json] [--prompt "text"] [--max-new N]
//                   [--stop <id> ...] [--stop-string "str" ...]
//                   [--fp32] [--prefill-chunk N]
//                   [--config config.json] [--max-seq-len N]
//                   [--chat] [--system "text"] [--role user]
//                   [--add-bos | --no-add-bos]
//                   [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//                   [--repetition-penalty R] [--seed S]
//
// Base Config is lfm2_5_24b_a2b() (40 layers, 10 attention + 30 short-conv,
// 64 experts top-4 after 2 dense layers); --tiny forces tiny_test() dims so
// the default run needs no download. --config overlays an HF config.json
// onto the selected base (present keys win, missing keys keep defaults).
//
// Sampling defaults to temperature=0.1, top_k=50, repetition_penalty=1.05
// (top_p=1.0, min_p=0.0); --temperature 0 restores greedy decode. Every
// sampling flag overrides the default; seed defaults to 0 so runs repeat
// unless --seed is given.
//
// Default compute/storage is bfloat16 (the GGUF checkpoint dtype); --fp32
// opts into FP32 for comparison. --prefill-chunk N caps prefill
// activations to O(N·dim) (0 = single-shot); default is
// Config::prefill_chunk_size.
//
// Chat: --chat wraps --prompt as one `role` (default "user") turn via the
// LFM2.5-Instruct ChatML renderer (open assistant header appended);
// --system prepends a system turn (implies chat). Raw mode (default)
// encodes --prompt directly.
//
// BOS: --add-bos (default) ensures exactly one leading BOS id (tokenizer
// bos id, else Config bos); --no-add-bos strips it. The chat path never
// double-counts BOS (encode_chatml keeps the exactly-one-BOS invariant).
//
// Stopping: by default the Instruct stop <|im_end|> (tokenizer eos, else
// Config eos); --stop appends extra ids. --stop-string appends decoded-text
// stop strings (repeatable, default includes "<|im_end|>"); generation is
// truncated before the earliest match.
template <class Scalar>
int run_model(KokkosDNN::LFM25::Config cfg, const std::string& wpath,
              bool tiny, const std::string& tok_path, const std::string& prompt,
              int max_new, const std::vector<int>& extra_stops,
              int prefill_chunk_override, bool chat_mode,
              const std::string& system_prompt, const std::string& role,
              bool want_bos, const std::vector<std::string>& stop_strings,
              KokkosDNN::SampleOptions samp_opt) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::LFM25::Weights<Exec, Scalar> w;
  if (tiny || wpath.empty()) {
    KokkosDNN::LFM25::random_init(cfg, w);
    std::printf("[%srandom weights] dim=%d layers=%d (attn=%d conv=%d) "
                "experts=%d top_k=%d\n",
                tiny ? "tiny " : "", cfg.dim, cfg.n_layers,
                cfg.num_attention_layers(), cfg.num_conv_layers(),
                cfg.n_experts, cfg.top_k);
  } else {
    KokkosDNN::LFM25::load_gguf(cfg, wpath, w);
    std::printf("[loaded %s]\n", wpath.c_str());
  }

  std::vector<int> prompt_ids;
  std::vector<int> stops;
  const bool use_chat = chat_mode || !system_prompt.empty();
  if (!tok_path.empty()) {
    KokkosDNN::LFM25::Tokenizer tok(tok_path);
    if (use_chat) {
      auto msgs = KokkosDNN::LFM25::make_chat_messages(
          prompt, system_prompt, role.empty() ? "user" : role);
      std::string rendered =
          KokkosDNN::LFM25::render_lfm25_chatml(msgs, true);
      std::printf("[chatml] %s", rendered.c_str());
      std::printf("\n");
      prompt_ids =
          KokkosDNN::LFM25::encode_chatml(tok, msgs, true, want_bos);
    } else {
      prompt_ids = KokkosDNN::LFM25::ensure_bos(tok.encode(prompt, want_bos),
                                               tok.bos(), want_bos);
    }
    std::printf("prompt tokens: %zu\n", prompt_ids.size());
    stops = {tok.eos(), tok.eot()};
  } else {
    // byte-fallback prompt (no tokenizer file): deterministic ids.
    std::string text = prompt;
    if (use_chat) {
      auto msgs = KokkosDNN::LFM25::make_chat_messages(
          prompt, system_prompt, role.empty() ? "user" : role);
      text = KokkosDNN::LFM25::render_lfm25_chatml(msgs, true);
      std::printf("[chatml, no tokenizer] %s\n", text.c_str());
    }
    if (want_bos) prompt_ids.push_back(cfg.bos % cfg.vocab_size);
    for (unsigned char c : text) prompt_ids.push_back(c % cfg.vocab_size);
    std::printf("[no tokenizer] byte-fallback prompt tokens: %zu\n",
                prompt_ids.size());
    stops = {cfg.eos};
  }
  stops.insert(stops.end(), extra_stops.begin(), extra_stops.end());
  std::printf("stop ids:");
  for (auto id : stops) std::printf(" %d", id);
  std::printf("\n");
  if (prefill_chunk_override >= 0)
    cfg.prefill_chunk_size = prefill_chunk_override;
  std::printf("prefill chunk: %d\n", cfg.prefill_chunk_size);
  try {
    cfg.validate();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: invalid config: %s\n", e.what());
    return 1;
  }
  {
    const double mib =
        (double)cfg.kv_cache_bytes(sizeof(Scalar)) / (1024.0 * 1024.0);
    std::printf("kv cache: dense fp, %d attention layer(s), full-window %.1f "
                "MiB\n",
                cfg.num_attention_layers(), mib);
  }
  if (prompt_ids.size() > (size_t)cfg.max_seq_len) {
    std::fprintf(stderr,
                 "error: prompt is %zu tokens but max_seq_len is %d "
                 "(raise --max-seq-len or pass --tokenizer)\n",
                 prompt_ids.size(), cfg.max_seq_len);
    return 1;
  }

  KokkosDNN::LFM25::Model<Exec, Scalar> model(cfg, w);
  std::printf("sampling: temp=%g top_k=%d top_p=%g min_p=%g rep_pen=%g "
              "seed=%llu%s\n",
              (double)samp_opt.temperature, samp_opt.top_k,
              (double)samp_opt.top_p, (double)samp_opt.min_p,
              (double)samp_opt.repetition_penalty,
              (unsigned long long)samp_opt.seed,
              samp_opt.is_greedy() ? " [greedy]" : "");
  KokkosDNN::HostRNG rng(samp_opt.seed);
  std::vector<int> out;
  if (!tok_path.empty()) {
    KokkosDNN::LFM25::Tokenizer stok(tok_path);
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
    KokkosDNN::LFM25::Tokenizer tok(tok_path);
    std::printf("decoded: %s\n", tok.decode(out).c_str());
    // Stop-strings are checked on decoded generated text (prompt
    // excluded so prompt content cannot false-trigger): truncate the
    // generation before the earliest match.
    std::vector<std::string> sstrings = stop_strings.empty()
                                            ? KokkosDNN::LFM25::default_stop_strings()
                                            : stop_strings;
    std::vector<int> gen_ids(out.begin() + (long)prompt_ids.size(), out.end());
    std::string gen_text = tok.decode(gen_ids);
    std::string cut =
        KokkosDNN::LFM25::truncate_at_stop_strings(gen_text, sstrings);
    if (cut.size() != gen_text.size()) {
      std::string hit;
      KokkosDNN::LFM25::find_stop_string(gen_text, sstrings, &hit);
      std::printf("stop-string hit: '%s'\n", hit.c_str());
    }
    std::printf("decoded (stop-strings applied): %s\n", cut.c_str());
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
    std::string wpath;
    std::string tok_path, prompt = "Hello, I am";
    std::string config_path;
    int max_new = 16;
    bool tiny   = true;
    bool fp32   = false;
    std::vector<int> extra_stops;
    int prefill_chunk_override = -1;
    int max_seq_len_override   = -1;
    bool chat_mode             = false;
    std::string system_prompt;
    std::string role = "user";
    bool want_bos    = true;
    std::vector<std::string> stop_strings;
    // LFM2.5 driver sampling defaults.
    KokkosDNN::SampleOptions samp_opt;
    samp_opt.temperature         = 0.1f;
    samp_opt.top_k               = 50;
    samp_opt.repetition_penalty  = 1.05f;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--weights" && i + 1 < argc) {
        wpath = argv[++i];
        tiny  = false;
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
    KokkosDNN::LFM25::Config cfg = KokkosDNN::LFM25::Config::lfm2_5_24b_a2b();
    std::printf("[model lfm2.5-24b-a2b]\n");
    if (tiny) {
      cfg               = KokkosDNN::LFM25::Config::tiny_test();
      cfg.max_seq_len   = 64;
    }
    if (!config_path.empty()) {
      try {
        cfg = KokkosDNN::LFM25::Config::from_json_file(config_path, cfg);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --config '%s': %s\n",
                     config_path.c_str(), e.what());
        Kokkos::finalize();
        return 1;
      }
      std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d "
                  "moe_inter=%d experts=%d top_k=%d dense=%d vocab=%d eps=%g "
                  "theta=%g max_seq=%d tie=%d\n",
                  config_path.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads,
                  cfg.n_kv_heads, cfg.ffn_dim, cfg.moe_intermediate,
                  cfg.n_experts, cfg.top_k, cfg.n_dense_layers, cfg.vocab_size,
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
      rc = run_model<float>(cfg, wpath, tiny, tok_path, prompt, max_new,
                            extra_stops, prefill_chunk_override, chat_mode,
                            system_prompt, role, want_bos, stop_strings,
                            samp_opt);
    } else {
      std::printf("[dtype bfloat16]\n");
      rc = run_model<KokkosDNN::LFM25::DefaultScalar>(
          cfg, wpath, tiny, tok_path, prompt, max_new, extra_stops,
          prefill_chunk_override, chat_mode, system_prompt, role, want_bos,
          stop_strings, samp_opt);
    }
    Kokkos::finalize();
    return rc;
  }
}
