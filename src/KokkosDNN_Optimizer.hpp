// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_OPTIMIZER_HPP
#define KOKKOSDNN_OPTIMIZER_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// SGD: p -= lr * g. Matrix and vector overloads; in-place on p.
template <class ExecSpace, class Scalar>
void sgd_step(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& param,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& grad,
    float lr, const std::string& label = "KokkosDNN::sgd_step") {
  if (param.extent(0) != grad.extent(0) || param.extent(1) != grad.extent(1))
    throw std::invalid_argument("sgd_step: dimension mismatch");
  const size_t R = param.extent(0), C = param.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        param(i, j) = static_cast<Scalar>(static_cast<float>(param(i, j)) -
                                          lr * static_cast<float>(grad(i, j)));
      });
}

template <class ExecSpace, class Scalar>
void sgd_step(
    const Kokkos::View<Scalar*, ExecSpace>& param,
    const Kokkos::View<Scalar*, ExecSpace>& grad, float lr,
    const std::string& label = "KokkosDNN::sgd_step_vec") {
  if (param.extent(0) != grad.extent(0))
    throw std::invalid_argument("sgd_step: dimension mismatch");
  const size_t N = param.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        param(i) = static_cast<Scalar>(static_cast<float>(param(i)) -
                                       lr * static_cast<float>(grad(i)));
      });
}

// Adam (Kingma & Ba 2014) with bias correction, per-parameter first/second
// moments m/v kept by the caller (zero-initialized). t is the 1-based step
// counter shared across all parameters (increment once per epoch for
// full-batch training). Standard defaults beta1=0.9, beta2=0.999, eps=1e-8.
template <class ExecSpace, class Scalar>
void adam_step(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& param,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& grad,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& m,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& v, float lr,
    float beta1, float beta2, float eps, long step,
    const std::string& label = "KokkosDNN::adam_step") {
  if (param.extent(0) != grad.extent(0) || param.extent(1) != grad.extent(1) ||
      param.extent(0) != m.extent(0) || param.extent(1) != m.extent(1) ||
      param.extent(0) != v.extent(0) || param.extent(1) != v.extent(1))
    throw std::invalid_argument("adam_step: dimension mismatch");
  if (step < 1) throw std::invalid_argument("adam_step: step must be >= 1");
  const size_t R = param.extent(0), C = param.extent(1);
  if (R == 0 || C == 0) return;
  const float bc1 = 1.0f - Kokkos::powf(beta1, static_cast<float>(step));
  const float bc2 = 1.0f - Kokkos::powf(beta2, static_cast<float>(step));
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float g  = static_cast<float>(grad(i, j));
        float mm = beta1 * static_cast<float>(m(i, j)) + (1.0f - beta1) * g;
        float vv = beta2 * static_cast<float>(v(i, j)) + (1.0f - beta2) * g * g;
        m(i, j)  = static_cast<Scalar>(mm);
        v(i, j)  = static_cast<Scalar>(vv);
        float mhat = mm / bc1, vhat = vv / bc2;
        param(i, j) = static_cast<Scalar>(static_cast<float>(param(i, j)) -
                                          lr * mhat / (Kokkos::sqrtf(vhat) + eps));
      });
}

template <class ExecSpace, class Scalar>
void adam_step(
    const Kokkos::View<Scalar*, ExecSpace>& param,
    const Kokkos::View<Scalar*, ExecSpace>& grad,
    const Kokkos::View<Scalar*, ExecSpace>& m,
    const Kokkos::View<Scalar*, ExecSpace>& v, float lr, float beta1,
    float beta2, float eps, long step,
    const std::string& label = "KokkosDNN::adam_step_vec") {
  if (param.extent(0) != grad.extent(0) || param.extent(0) != m.extent(0) ||
      param.extent(0) != v.extent(0))
    throw std::invalid_argument("adam_step: dimension mismatch");
  if (step < 1) throw std::invalid_argument("adam_step: step must be >= 1");
  const size_t N = param.extent(0);
  if (N == 0) return;
  const float bc1 = 1.0f - Kokkos::powf(beta1, static_cast<float>(step));
  const float bc2 = 1.0f - Kokkos::powf(beta2, static_cast<float>(step));
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        float g  = static_cast<float>(grad(i));
        float mm = beta1 * static_cast<float>(m(i)) + (1.0f - beta1) * g;
        float vv = beta2 * static_cast<float>(v(i)) + (1.0f - beta2) * g * g;
        m(i)     = static_cast<Scalar>(mm);
        v(i)     = static_cast<Scalar>(vv);
        float mhat = mm / bc1, vhat = vv / bc2;
        param(i) = static_cast<Scalar>(static_cast<float>(param(i)) -
                                       lr * mhat / (Kokkos::sqrtf(vhat) + eps));
      });
}

}  // namespace KokkosDNN

#endif
