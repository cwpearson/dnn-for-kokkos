// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Trains the XOR MLP (2 -> hidden tanh -> 1) with full-batch Adam and
// prints the final truth table. Usage:
//   xor_train [--hidden N] [--epochs N] [--lr X] [--seed N]
//             [--print-every N]
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "KokkosDNN_XorMLP.hpp"

namespace {
struct Args {
  size_t hidden = 8;
  long epochs = 5000;
  float lr = 0.05f;
  uint64_t seed = 0;
  bool seed_given = false;
  long print_every = 500;
};

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&](long& n) {
      if (++i >= argc) return false;
      n = std::stol(argv[i]);
      return true;
    };
    if (s == "--hidden") {
      long v = 0;
      if (!need(v) || v < 1) return false;
      a.hidden = static_cast<size_t>(v);
    } else if (s == "--epochs") {
      if (!need(a.epochs) || a.epochs < 1) return false;
    } else if (s == "--lr") {
      if (++i >= argc) return false;
      a.lr = std::stof(argv[i]);
      if (!(a.lr > 0)) return false;
    } else if (s == "--seed") {
      long v = 0;
      if (!need(v) || v < 0) return false;
      a.seed = static_cast<uint64_t>(v);
      a.seed_given = true;
    } else if (s == "--print-every") {
      if (!need(a.print_every) || a.print_every < 0) return false;
    } else if (s == "--help" || s == "-h") {
      return false;
    } else {
      std::printf("unknown arg: %s\n", argv[i]);
      return false;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    std::printf(
        "usage: xor_train [--hidden N] [--epochs N] [--lr X] [--seed N] "
        "[--print-every N]\n");
    return 2;
  }
  Kokkos::ScopeGuard guard(argc, argv);
  using Exec = Kokkos::DefaultExecutionSpace;
  using Scalar = float;

  if (!args.seed_given) {
    std::random_device rd;
    args.seed = (uint64_t(rd()) << 32) | uint64_t(rd());
  }

  auto data = KokkosDNN::Xor::make_dataset<Exec, Scalar>();
  auto mlp = KokkosDNN::Xor::create_mlp<Exec, Scalar>(args.hidden, args.seed);
  auto ws =
      KokkosDNN::Xor::create_workspace<Exec, Scalar>(4, args.hidden);

  float loss = 0;
  for (long e = 1; e <= args.epochs; ++e) {
    loss = KokkosDNN::Xor::train_epoch<Exec, Scalar>(
        mlp, data.X, data.T, ws, e, args.lr);
    if (args.print_every > 0 &&
        (e % args.print_every == 0 || e == args.epochs))
      std::printf("epoch %ld loss %.6f\n", e, loss);
  }
  Kokkos::fence();

  auto probs = KokkosDNN::Xor::predict_probs<Exec, Scalar>(mlp, data.X, ws);
  int n_correct = 0;
  std::printf("x1 x2 | prob -> pred (want)\n");
  const int want[4] = {0, 1, 1, 0};
  const int xin[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  for (int i = 0; i < 4; ++i) {
    float p = static_cast<float>(probs(i, 0));
    int pred = p >= 0.5f ? 1 : 0;
    n_correct += (pred == want[i]);
    std::printf("%d  %d  | %.4f -> %d (%d)\n", xin[i][0], xin[i][1], p,
                pred, want[i]);
  }
  std::printf("final loss %.6f, correct %d/4\n", loss, n_correct);
  return n_correct == 4 ? 0 : 1;
}
