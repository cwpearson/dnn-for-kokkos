// End-to-end XOR test: a tiny 2->8(tanh)->1 MLP trained with full-batch
// Adam must learn the truth table. Runs on DefaultExecutionSpace (Serial
// in CPU CI) in well under a second of kernel time plus Kokkos launch
// overhead; kept deterministic via a fixed init seed.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>

#include "KokkosDNN.hpp"
#include "KokkosDNN_XorMLP.hpp"

using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = float;

// Library-level gradient sanity checks (catch sign/scale regressions in the
// new training ops without paying for a training loop).
TEST(XorOps, ActivationGradients) {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> y("y", 1, 3),
      dy("dy", 1, 3), dx("dx", 1, 3);
  auto hy = Kokkos::create_mirror_view(y);
  auto hdy = Kokkos::create_mirror_view(dy);
  // sigmoid(0) = 0.5 -> grad 0.25 * dy; tanh(0) = 0 -> grad 1 * dy.
  hy(0, 0) = 0.5f;
  hy(0, 1) = 0.0f;
  hy(0, 2) = 1.0f;
  hdy(0, 0) = 1.0f;
  hdy(0, 1) = 2.0f;
  hdy(0, 2) = -1.0f;
  Kokkos::deep_copy(y, hy);
  Kokkos::deep_copy(dy, hdy);
  KokkosDNN::sigmoid_backward<Exec, Scalar>(y, dy, dx);
  Kokkos::fence();
  auto hs = Kokkos::create_mirror_view(dx);
  Kokkos::deep_copy(hs, dx);
  EXPECT_NEAR(hs(0, 0), 0.25f, 1e-6f);
  EXPECT_NEAR(hs(0, 1), 0.0f, 1e-6f);   // s*(1-s) = 0 at s=0
  EXPECT_NEAR(hs(0, 2), -0.0f, 1e-6f);  // s*(1-s) = 0 at s=1
  KokkosDNN::tanh_backward<Exec, Scalar>(y, dy, dx);
  Kokkos::fence();
  Kokkos::deep_copy(hs, dx);
  EXPECT_NEAR(hs(0, 0), 0.75f, 1e-6f);  // (1-0.25)*1
  EXPECT_NEAR(hs(0, 1), 2.0f, 1e-6f);
  EXPECT_NEAR(hs(0, 2), -0.0f, 1e-6f);  // (1-1)*dy
}

TEST(XorOps, DenseBackwardBiasIsColumnSum) {
  // dY = [[1,2],[3,4]] -> db = [4,6]; X/W contents only shape the other
  // outputs, which this test does not check.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> dY("dY", 2, 2),
      X("X", 2, 3), W("W", 2, 3), dX("dX", 2, 3), dW("dW", 2, 3);
  Kokkos::View<Scalar*, Exec> db("db", 2);
  auto h = Kokkos::create_mirror_view(dY);
  h(0, 0) = 1; h(0, 1) = 2; h(1, 0) = 3; h(1, 1) = 4;
  Kokkos::deep_copy(dY, h);
  KokkosDNN::dense_backward<Exec, Scalar>(dY, X, W, dX, dW, db);
  Kokkos::fence();
  auto hb = Kokkos::create_mirror_view(db);
  Kokkos::deep_copy(hb, db);
  EXPECT_NEAR(hb(0), 4.0f, 1e-6f);
  EXPECT_NEAR(hb(1), 6.0f, 1e-6f);
}

TEST(XorOps, BceLossOfZeroLogitsIsLog2) {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> z("z", 2, 1),
      t("t", 2, 1);
  auto ht = Kokkos::create_mirror_view(t);
  ht(0, 0) = 0; ht(1, 0) = 1;
  Kokkos::deep_copy(t, ht);
  // Logits are zero-initialized: loss = log(2) regardless of targets.
  float loss = KokkosDNN::bce_with_logits_loss<Exec, Scalar>(z, t);
  EXPECT_NEAR(loss, std::log(2.0f), 1e-6f);
}

TEST(Xor, TrainsToTruthTable) {
  constexpr size_t kHidden = 8;
  constexpr long kEpochs = 5000;
  auto data = KokkosDNN::Xor::make_dataset<Exec, Scalar>();
  auto mlp = KokkosDNN::Xor::create_mlp<Exec, Scalar>(kHidden, /*seed=*/0);
  auto ws = KokkosDNN::Xor::create_workspace<Exec, Scalar>(4, kHidden);

  float loss = 0;
  for (long e = 1; e <= kEpochs; ++e)
    loss = KokkosDNN::Xor::train_epoch<Exec, Scalar>(mlp, data.X, data.T, ws,
                                                    e, 0.05f);
  Kokkos::fence();
  EXPECT_LT(loss, 0.05f) << "XOR training did not converge";

  auto probs = KokkosDNN::Xor::predict_probs<Exec, Scalar>(mlp, data.X, ws);
  const float want[4] = {0, 1, 1, 0};
  for (int i = 0; i < 4; ++i) {
    float p = static_cast<float>(probs(i, 0));
    EXPECT_NEAR(p, want[i], 0.2f) << "XOR case " << i;
    EXPECT_EQ(p >= 0.5f ? 1 : 0, static_cast<int>(want[i]))
        << "XOR misclassified case " << i;
  }
}
