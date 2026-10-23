// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Two-layer MLP for XOR, built only from KokkosDNN training ops.
//
// Architecture: X[B,2] -> dense(W1[H,2],b1) -> tanh -> dense(W2[1,H],b2)
// -> logits[B,1] -> BCE-with-logits. Full-batch Adam. All tensors live in
// ExecSpace; init runs on host mirrors with a seeded mt19937 (Xavier
// uniform weights, zero biases) so runs are reproducible.
#ifndef KOKKOSDNN_XOR_MLP_HPP
#define KOKKOSDNN_XOR_MLP_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <random>

#include "KokkosDNN.hpp"

namespace KokkosDNN::Xor {

// Canonical XOR dataset: rows of X are the four input pairs, T the labels.
template <class ExecSpace, class Scalar = float>
struct Dataset {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> X;  // [4,2]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> T;  // [4,1]
};

template <class ExecSpace, class Scalar = float>
Dataset<ExecSpace, Scalar> make_dataset() {
  Dataset<ExecSpace, Scalar> d;
  d.X = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_X", 4, 2);
  d.T = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_T", 4, 1);
  auto hx = Kokkos::create_mirror_view(d.X);
  auto ht = Kokkos::create_mirror_view(d.T);
  hx(0, 0) = 0; hx(0, 1) = 0; ht(0, 0) = 0;
  hx(1, 0) = 0; hx(1, 1) = 1; ht(1, 0) = 1;
  hx(2, 0) = 1; hx(2, 1) = 0; ht(2, 0) = 1;
  hx(3, 0) = 1; hx(3, 1) = 1; ht(3, 0) = 0;
  Kokkos::deep_copy(d.X, hx);
  Kokkos::deep_copy(d.T, ht);
  return d;
}

template <class ExecSpace, class Scalar = float>
struct MLP {
  size_t hidden = 0;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> W1;  // [H,2]
  Kokkos::View<Scalar*, ExecSpace> b1;                        // [H]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> W2;  // [1,H]
  Kokkos::View<Scalar*, ExecSpace> b2;                        // [1]
  // Adam moments (zero-initialized), same shapes as the parameters.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mW1, vW1, mW2, vW2;
  Kokkos::View<Scalar*, ExecSpace> mb1, vb1, mb2, vb2;
  DNNHandle<ExecSpace> handle;
};

// Scratch + gradient buffers for one batch size / hidden width. Allocate
// once and reuse across epochs (no per-epoch allocation in the loop).
template <class ExecSpace, class Scalar = float>
struct Workspace {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> Z1, A1;  // [B,H]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> logits;  // [B,1]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dlogits;  // [B,1]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dA1, dZ1;  // [B,H]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dX;        // [B,2]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dW1;  // [H,2]
  Kokkos::View<Scalar*, ExecSpace> db1;                        // [H]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dW2;  // [1,H]
  Kokkos::View<Scalar*, ExecSpace> db2;                        // [1]
};

template <class ExecSpace, class Scalar = float>
Workspace<ExecSpace, Scalar> create_workspace(size_t batch, size_t hidden) {
  Workspace<ExecSpace, Scalar> w;
  w.Z1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_Z1", batch, hidden);
  w.A1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_A1", batch, hidden);
  w.logits = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_logits", batch, 1);
  w.dlogits = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_dlogits", batch, 1);
  w.dA1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_dA1", batch, hidden);
  w.dZ1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_dZ1", batch, hidden);
  w.dX = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_dX", batch, 2);
  w.dW1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_dW1", hidden, 2);
  w.db1 = Kokkos::View<Scalar*, ExecSpace>("xor_db1", hidden);
  w.dW2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_dW2", 1, hidden);
  w.db2 = Kokkos::View<Scalar*, ExecSpace>("xor_db2", 1);
  return w;
}

namespace detail {
inline float xavier_limit(size_t fan_in, size_t fan_out) {
  return std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
}
}  // namespace detail

template <class ExecSpace, class Scalar = float>
MLP<ExecSpace, Scalar> create_mlp(size_t hidden, uint64_t seed = 0) {
  MLP<ExecSpace, Scalar> m;
  m.hidden = hidden;
  m.W1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_W1", hidden, 2);
  m.b1 = Kokkos::View<Scalar*, ExecSpace>("xor_b1", hidden);
  m.W2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_W2", 1, hidden);
  m.b2 = Kokkos::View<Scalar*, ExecSpace>("xor_b2", 1);
  m.mW1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_mW1", hidden, 2);
  m.vW1 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_vW1", hidden, 2);
  m.mb1 = Kokkos::View<Scalar*, ExecSpace>("xor_mb1", hidden);
  m.vb1 = Kokkos::View<Scalar*, ExecSpace>("xor_vb1", hidden);
  m.mW2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_mW2", 1, hidden);
  m.vW2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("xor_vW2", 1, hidden);
  m.mb2 = Kokkos::View<Scalar*, ExecSpace>("xor_mb2", 1);
  m.vb2 = Kokkos::View<Scalar*, ExecSpace>("xor_vb2", 1);

  std::mt19937 eng(seed);
  auto fill_xavier = [&](auto& host, float limit) {
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (size_t i = 0; i < host.extent(0); ++i)
      for (size_t j = 0; j < host.extent(1); ++j)
        host(i, j) = static_cast<Scalar>(dist(eng));
  };
  auto hW1 = Kokkos::create_mirror_view(m.W1);
  auto hW2 = Kokkos::create_mirror_view(m.W2);
  fill_xavier(hW1, detail::xavier_limit(2, hidden));
  fill_xavier(hW2, detail::xavier_limit(hidden, 1));
  Kokkos::deep_copy(m.W1, hW1);
  Kokkos::deep_copy(m.W2, hW2);
  // Biases and Adam moments stay zero-initialized by the View constructor.
  return m;
}

// Forward only (inference or the forward half of a training step).
template <class ExecSpace, class Scalar = float>
void forward(const MLP<ExecSpace, Scalar>& m,
             const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
             Workspace<ExecSpace, Scalar>& ws) {
  dense_forward(m.handle, X, m.W1, m.b1, ws.Z1);
  tanh_forward<ExecSpace, Scalar>(ws.Z1, ws.A1);
  dense_forward(m.handle, ws.A1, m.W2, m.b2, ws.logits);
}

// One full-batch Adam step; returns the pre-update mean BCE loss. step is
// the 1-based Adam counter (increment once per epoch).
template <class ExecSpace, class Scalar = float>
float train_epoch(MLP<ExecSpace, Scalar>& m,
                  const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
                  const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& T,
                  Workspace<ExecSpace, Scalar>& ws, long step, float lr = 0.05f,
                  float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
  forward(m, X, ws);
  float loss = bce_with_logits_loss<ExecSpace, Scalar>(ws.logits, T);
  bce_with_logits_backward<ExecSpace, Scalar>(ws.logits, T, ws.dlogits);
  dense_backward<ExecSpace, Scalar>(ws.dlogits, ws.A1, m.W2, ws.dA1, ws.dW2,
                                    ws.db2);
  tanh_backward<ExecSpace, Scalar>(ws.A1, ws.dA1, ws.dZ1);
  dense_backward<ExecSpace, Scalar>(ws.dZ1, X, m.W1, ws.dX, ws.dW1, ws.db1);
  adam_step<ExecSpace, Scalar>(m.W1, ws.dW1, m.mW1, m.vW1, lr, beta1, beta2,
                               eps, step);
  adam_step<ExecSpace, Scalar>(m.b1, ws.db1, m.mb1, m.vb1, lr, beta1, beta2,
                               eps, step);
  adam_step<ExecSpace, Scalar>(m.W2, ws.dW2, m.mW2, m.vW2, lr, beta1, beta2,
                               eps, step);
  adam_step<ExecSpace, Scalar>(m.b2, ws.db2, m.mb2, m.vb2, lr, beta1, beta2,
                               eps, step);
  return loss;
}

// Sigmoid probabilities for the batch in ws.logits (host mirror out).
template <class ExecSpace, class Scalar = float>
Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> predict_probs(
    const MLP<ExecSpace, Scalar>& m,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    Workspace<ExecSpace, Scalar>& ws) {
  forward(m, X, ws);
  auto h = Kokkos::create_mirror_view(ws.logits);
  Kokkos::deep_copy(h, ws.logits);
  auto probs =
      Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace>("probs",
                                                                     h.extent(0), 1);
  for (size_t i = 0; i < h.extent(0); ++i) {
    float z = static_cast<float>(h(i, 0));
    probs(i, 0) = static_cast<Scalar>(1.0f / (1.0f + std::exp(-z)));
  }
  return probs;
}

}  // namespace KokkosDNN::Xor

#endif
