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

template <class Scalar>
void check_multi_stop() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};

  // Full run: default Instruct stops are out of tiny vocab, so no early stop.
  auto full = model.generate(prompt, 4, std::vector<int>{});
  ASSERT_EQ(full.size(), prompt.size() + 4);

  // Re-run stopping on the known first generated token: must stop after
  // exactly one new token (stop token included).
  const int first = full[prompt.size()];
  KokkosDNN::Llama::Model<Exec, Scalar> model2(cfg, w);
  auto early =
      model2.generate(prompt, 4, std::vector<int>{first, first});  // dupes ok
  ASSERT_EQ(early.size(), prompt.size() + 1);
  EXPECT_EQ(early.back(), first);

  // Non-triggering stop set: full length again.
  KokkosDNN::Llama::Model<Exec, Scalar> model3(cfg, w);
  auto no_stop = model3.generate(prompt, 4, std::vector<int>{cfg.vocab_size + 7});
  EXPECT_EQ(no_stop, full);

  // Back-compat single-eos overload agrees with the vector overload.
  KokkosDNN::Llama::Model<Exec, Scalar> model4(cfg, w);
  auto via_int = model4.generate(prompt, 4, first);
  EXPECT_EQ(via_int, early);
}

TEST(LlamaGenerateMultiStop, Float) { check_multi_stop<float>(); }

TEST(LlamaGenerateMultiStop, BHalf) {
  check_multi_stop<Kokkos::Experimental::bhalf_t>();
}
