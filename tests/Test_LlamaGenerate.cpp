// End-to-end tiny Llama: random weights, prefill + greedy decode.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    auto cfg = KokkosDNN::Llama::Config::tiny_test();
    KokkosDNN::Llama::Weights<Exec> w;
    KokkosDNN::Llama::random_init(cfg, w, 1234);
    KokkosDNN::Llama::Model<Exec> model(cfg, w);
    std::vector<int> prompt = {1, 2, 3, 4};
    auto out = model.generate(prompt, 4);
    if (out.size() != prompt.size() + 4) {
      std::printf("FAIL llama_generate size %zu\n", out.size());
      return 1;
    }
    for (auto id : out)
      if (id < 0 || id >= cfg.vocab_size) {
        std::printf("FAIL llama_generate id %d\n", id);
        return 1;
      }
    // Determinism: rebuild and compare.
    KokkosDNN::Llama::Model<Exec> model2(cfg, w);
    auto out2 = model2.generate(prompt, 4);
    if (out != out2) {
      std::printf("FAIL llama_generate nondeterministic\n");
      return 1;
    }
    std::printf("PASS test_llama_generate ids:");
    for (auto id : out) std::printf(" %d", id);
    std::printf("\n");
  }
  Kokkos::finalize();
  return 0;
}
