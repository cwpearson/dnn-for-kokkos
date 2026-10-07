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
//
// FP32 8B needs ~32 GiB; RTX 3090 (24 GiB) cannot hold it — use --tiny or
// a big-GPU/CPU build for real checkpoints.
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    std::vector<std::string> wfiles;
    std::string tok_path, prompt = "Hello, I am";
    int max_new = 16;
    bool tiny = true;

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
      } else if (a == "--tiny") {
        tiny = true;
      }
    }

    KokkosDNN::print_configuration(std::cout);
    KokkosDNN::Llama::Config cfg = KokkosDNN::Llama::Config::llama_3_3_8b();
    if (tiny) {
      cfg = KokkosDNN::Llama::Config::tiny_test();
      cfg.max_seq_len = 64;
    }
    KokkosDNN::Llama::Weights<Exec> w;
    if (tiny || wfiles.empty()) {
      KokkosDNN::Llama::random_init(cfg, w);
      std::printf("[tiny random weights] dim=%d layers=%d\n", cfg.dim,
                  cfg.n_layers);
    } else {
      KokkosDNN::Llama::load_safetensors(cfg, wfiles, w);
      std::printf("[loaded %zu shard(s)]\n", wfiles.size());
    }

    std::vector<int> prompt_ids;
    if (!tok_path.empty()) {
      KokkosDNN::Llama::Tokenizer tok(tok_path);
      prompt_ids = tok.encode(prompt);
      std::printf("prompt tokens: %zu\n", prompt_ids.size());
    } else {
      // byte-fallback prompt (no tokenizer file): deterministic ids
      prompt_ids.push_back(cfg.bos_token % cfg.vocab_size);
      for (unsigned char c : prompt)
        prompt_ids.push_back(c % cfg.vocab_size);
      std::printf("[no tokenizer] byte-fallback prompt tokens: %zu\n",
                  prompt_ids.size());
    }

    KokkosDNN::Llama::Model<Exec> model(cfg, w);
    auto out = model.generate(prompt_ids, max_new);
    std::printf("generated %zu ids:", out.size());
    for (auto id : out) std::printf(" %d", id);
    std::printf("\n");
    if (!tok_path.empty()) {
      KokkosDNN::Llama::Tokenizer tok(tok_path);
      std::printf("decoded: %s\n", tok.decode(out).c_str());
    }
    Kokkos::fence();
  }
  Kokkos::finalize();
  return 0;
}
