// End-to-end tiny Llama: random weights, prefill + greedy decode.
// Exercises both bfloat16 (default) and fp32 scalar paths.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

template <class Scalar>
void check_generate() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  auto out                = model.generate(prompt, 4);
  ASSERT_EQ(out.size(), prompt.size() + 4);
  for (auto id : out) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }
  // Determinism: rebuild and compare.
  KokkosDNN::Llama::Model<Exec, Scalar> model2(cfg, w);
  auto out2 = model2.generate(prompt, 4);
  EXPECT_EQ(out, out2);
}

TEST(LlamaGenerate, Float) { check_generate<float>(); }

TEST(LlamaGenerate, BHalf) { check_generate<Kokkos::Experimental::bhalf_t>(); }
