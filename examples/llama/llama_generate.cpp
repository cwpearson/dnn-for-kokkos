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
//
// Default compute/storage is bfloat16 (Llama 3.3 8B checkpoint dtype);
// --fp32 opts in. --prefill-chunk N caps prefill activations to
// O(N·dim) (0 = single-shot); default is Config::prefill_chunk_size.
//
// Default compute/storage is bfloat16 (Llama 3.3 8B checkpoint dtype);
// --fp32 opts into FP32 for comparison. BF16 8B needs ~16 GiB (FP32
// ~32 GiB); RTX 3090 (24 GiB) fits BF16 but not FP32 — use --tiny or
// a big-GPU/CPU build for real checkpoints.
//
// Stopping: by default the Instruct stop set {eos, eom, eot} from the
// tokenizer (or Config fallback without --tokenizer); --stop appends
// extra ids.
template <class Scalar>
int run_model(KokkosDNN::Llama::Config cfg,
              const std::vector<std::string>& wfiles, bool tiny,
              const std::string& tok_path, const std::string& prompt,
              int max_new, const std::vector<int>& extra_stops,
              int prefill_chunk_override) {
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
  if (!tok_path.empty()) {
    KokkosDNN::Llama::Tokenizer tok(tok_path);
    prompt_ids = tok.encode(prompt);
    std::printf("prompt tokens: %zu\n", prompt_ids.size());
    stops = {tok.eos(), tok.eom(), tok.eot()};
  } else {
    // byte-fallback prompt (no tokenizer file): deterministic ids
    prompt_ids.push_back(cfg.bos_token % cfg.vocab_size);
    for (unsigned char c : prompt) prompt_ids.push_back(c % cfg.vocab_size);
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

  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  auto out = model.generate(prompt_ids, max_new, stops);
  std::printf("generated %zu ids:", out.size());
  for (auto id : out) std::printf(" %d", id);
  std::printf("\n");
  if (!tok_path.empty()) {
    KokkosDNN::Llama::Tokenizer tok(tok_path);
    std::printf("decoded: %s\n", tok.decode(out).c_str());
  }
  Kokkos::fence();
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::vector<std::string> wfiles;
    std::string tok_path, prompt = "Hello, I am";
    int max_new = 16;
    bool tiny = true;
    bool fp32 = false;
    std::vector<int> extra_stops;
    int prefill_chunk_override = -1;

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
      } else if (a == "--prefill-chunk" && i + 1 < argc) {
        prefill_chunk_override = std::stoi(argv[++i]);
      } else if (a == "--fp32") {
        fp32 = true;
      }
    }

    KokkosDNN::print_configuration(std::cout);
    KokkosDNN::Llama::Config cfg = KokkosDNN::Llama::Config::llama_3_3_8b();
    if (tiny) {
      cfg = KokkosDNN::Llama::Config::tiny_test();
      cfg.max_seq_len = 64;
    }
    int rc;
    if (fp32) {
      std::printf("[dtype fp32]\n");
      rc = run_model<float>(cfg, wfiles, tiny, tok_path, prompt, max_new,
                            extra_stops, prefill_chunk_override);
    } else {
      std::printf("[dtype bfloat16]\n");
      rc = run_model<KokkosDNN::Llama::DefaultScalar>(cfg, wfiles, tiny,
                                                      tok_path, prompt, max_new,
                                                      extra_stops,
                                                      prefill_chunk_override);
    }
    Kokkos::finalize();
    return rc;
  }
}
