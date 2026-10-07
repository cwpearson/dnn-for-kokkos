// End-to-end tiny Llama: random weights, prefill + greedy decode.
// Exercises both bfloat16 (default) and fp32 scalar paths.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdio>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

template <class Scalar>
int run_generate(const char* tag) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  auto out = model.generate(prompt, 4);
  if (out.size() != prompt.size() + 4) {
    std::printf("FAIL llama_generate[%s] size %zu\n", tag, out.size());
    return 1;
  }
  for (auto id : out)
    if (id < 0 || id >= cfg.vocab_size) {
      std::printf("FAIL llama_generate[%s] id %d\n", tag, id);
      return 1;
    }
  // Determinism: rebuild and compare.
  KokkosDNN::Llama::Model<Exec, Scalar> model2(cfg, w);
  auto out2 = model2.generate(prompt, 4);
  if (out != out2) {
    std::printf("FAIL llama_generate[%s] nondeterministic\n", tag);
    return 1;
  }
  std::printf("PASS test_llama_generate[%s] ids:", tag);
  for (auto id : out) std::printf(" %d", id);
  std::printf("\n");
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (run_generate<float>("fp32")) return 1;
    if (run_generate<Kokkos::Experimental::bhalf_t>("bf16")) return 1;
  }
  Kokkos::finalize();
  return 0;
}
