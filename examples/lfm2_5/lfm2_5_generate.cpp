// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_CliOptions.hpp"
#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Model.hpp"
#include "KokkosDNN_LFM25Tokenizer.hpp"
#include "KokkosDNN_LFM25Weights.hpp"

// PoC driver: tiny random model by default; real weights via --weights.
//   lfm2_5_generate [--tiny] [--weights model.gguf]
//                   [--tokenizer tok.json] [--prompt "text"] [--max-new N]
//                   [--stop <id> ...] [--stop-string "str" ...]
//                   [--fp32] [--timings] [--prefill-chunk N]
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
// sampling flag overrides the default; seed is random unless --seed is
// given.
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
              KokkosDNN::SampleOptions samp_opt, bool want_timings) {
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
  // Per-phase timings (--timings): the driver otherwise prints none, so
  // throughput work ends up measuring whole-process wall time and
  // subtracting out the weight load. Captured at the sampling callback,
  // which fires once per generated token right after sampling: the gap from
  // generate() start to the first callback is prefill + first-token TTFT,
  // and the rest are decode steps.
  using Clock = std::chrono::steady_clock;
  const auto t_gen0 = Clock::now();
  auto since = [](Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };
  bool got_first = false;
  double t_first = 0.0, t_last = 0.0;
  std::vector<double> step_ms;
  auto mark_token = [&](int /*id*/) {
    const double t = since(t_gen0);
    if (!got_first) {
      got_first = true;
      t_first = t;
    } else {
      step_ms.push_back(t - t_last);
    }
    t_last = t;
  };
  if (!tok_path.empty()) {
    KokkosDNN::LFM25::Tokenizer stok(tok_path);
    std::printf("decoded (streaming): ");
    std::fflush(stdout);
    auto on_token = [&](int id) {
      mark_token(id);
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
      mark_token(id);
      std::printf(" %d", id);
      std::fflush(stdout);
    };
    out = model.generate(prompt_ids, max_new, stops, samp_opt, &rng, on_token);
    std::printf("\n");
  }
  std::printf("generated %zu ids:", out.size());
  for (auto id : out) std::printf(" %d", id);
  std::printf("\n");
  if (want_timings) {
    // Prompt tokens are forwarded by generate()'s prefill; every other id
    // is one decode step. The first sampled token is produced at the end of
    // prefill, so t_first is prefill + TTFT for that token.
    const size_t n_prompt = prompt_ids.size();
    const size_t n_new = out.size() > n_prompt ? out.size() - n_prompt : 0;
    double med = 0.0;
    if (!step_ms.empty()) {
      std::vector<double> s = step_ms;
      std::sort(s.begin(), s.end());
      med = s[s.size() / 2];
    }
    const double prefill_tok_s =
        t_first > 0.0 ? (double)n_prompt * 1e3 / t_first : 0.0;
    const double decode_ms = got_first ? (t_last - t_first) : 0.0;
    const double decode_tok_s =
        decode_ms > 0.0 ? (double)step_ms.size() * 1e3 / decode_ms : 0.0;
    std::fprintf(
        stderr,
        "[timings] prefill %zu tok in %.1f ms (%.1f tok/s) | decode %zu tok "
        "in %.1f ms (%.1f tok/s) | median step %.2f ms | total %.1f ms\n",
        n_prompt, t_first, prefill_tok_s, n_new, decode_ms, decode_tok_s, med,
        got_first ? t_last : since(t_gen0));
  }
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
    // LFM2.5 takes no --messages alias, so the shared parser runs with
    // messages disabled (--chat/--system/--role still shared).
    const KokkosDNN::Common::Cli::ParseAllow allow{true, false};
    KokkosDNN::Common::Cli::TextGenOptions o;
    // LFM2.5 driver sampling defaults.
    o.samp.temperature = 0.1f;
    o.samp.top_k = 50;
    o.samp.repetition_penalty = 1.05f;
    bool want_timings = false;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (KokkosDNN::Common::Cli::try_parse_shared(a, argv, i, argc, o,
                                                   allow)) {
        continue;
      } else if (a == "--timings") {
        want_timings = true;
      }
    }

    KokkosDNN::Common::Cli::resolve_random_seed(o);
    // Single-file GGUF weights: the shared parser collects --weights into
    // a list; the first entry wins (extra tokens were ignored before too).
    const std::string wpath = o.weights.empty() ? "" : o.weights.front();

    KokkosDNN::print_configuration(std::cout);
    KokkosDNN::LFM25::Config cfg = KokkosDNN::LFM25::Config::lfm2_5_24b_a2b();
    std::printf("[model lfm2.5-24b-a2b]\n");
    if (o.tiny) {
      cfg               = KokkosDNN::LFM25::Config::tiny_test();
      cfg.max_seq_len   = 64;
    }
    if (!o.config_path.empty()) {
      try {
        cfg = KokkosDNN::LFM25::Config::from_json_file(o.config_path, cfg);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "error: --config '%s': %s\n",
                     o.config_path.c_str(), e.what());
        Kokkos::finalize();
        return 1;
      }
      std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d "
                  "moe_inter=%d experts=%d top_k=%d dense=%d vocab=%d eps=%g "
                  "theta=%g max_seq=%d tie=%d\n",
                  o.config_path.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads,
                  cfg.n_kv_heads, cfg.ffn_dim, cfg.moe_intermediate,
                  cfg.n_experts, cfg.top_k, cfg.n_dense_layers, cfg.vocab_size,
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
      rc = run_model<float>(cfg, wpath, o.tiny, o.tokenizer_path, o.prompt,
                            o.max_new, o.extra_stops, o.prefill_chunk,
                            o.chat_mode, o.system_prompt, o.role, o.want_bos,
                            o.stop_strings, o.samp, want_timings);
    } else {
      std::printf("[dtype bfloat16]\n");
      rc = run_model<KokkosDNN::LFM25::DefaultScalar>(
          cfg, wpath, o.tiny, o.tokenizer_path, o.prompt, o.max_new,
          o.extra_stops, o.prefill_chunk, o.chat_mode, o.system_prompt,
          o.role, o.want_bos, o.stop_strings, o.samp, want_timings);
    }
    Kokkos::finalize();
    return rc;
  }
}
