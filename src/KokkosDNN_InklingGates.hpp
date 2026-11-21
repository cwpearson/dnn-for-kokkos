// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGGATES_HPP
#define KOKKOSDNN_INKLINGGATES_HPP

// Inkling2 loop gating elementwise ops (thinkingmachines/Inkling-2-Mini
// text decoder): Softcap, SoftcappedSwiGLU, the GatedOutputProjection
// elementwise gate multiply, and the rank-64 MessageGate. GEMM weights are
// supplied by the caller: this header never performs a matmul, only the
// elementwise math around them. All accumulation is FP32 with the result
// cast back to Scalar. Backend-portable: only parallel_for kernels; no
// backend-specific code.

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Softcap: y = tanh(x / s) * s, elementwise over [T, F]. Matches
// Inkling2LoopSoftcap (logit softcap 30.0, gate 4.0, up 25.0).
// In-place safe.
template <class ExecSpace, class Scalar>
void inkling_softcap(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    float softcap,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_softcap") {
  const size_t T = x.extent(0), F = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != F)
    throw std::invalid_argument("inkling_softcap: dimension mismatch");
  if (T == 0 || F == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, F}),
      KOKKOS_LAMBDA(size_t t, size_t f) {
        const float v = static_cast<float>(x(t, f)) / softcap;
        y(t, f) = static_cast<Scalar>(Kokkos::tanhf(v) * softcap);
      });
}

// SoftcappedSwiGLU: out = softcap(gate, gs) * sigmoid(gate_raw) * softcap(up,
// us). Note the sigmoid sees the RAW gate while the linear gate factor is
// softcapped (matches Inkling2LoopSoftcappedSwiGLU). gate, up, out: [T, F].
template <class ExecSpace, class Scalar>
void inkling_softcapped_swiglu(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up,
    float gate_softcap, float up_softcap,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::inkling_softcapped_swiglu") {
  const size_t T = gate.extent(0), F = gate.extent(1);
  if (up.extent(0) != T || up.extent(1) != F || out.extent(0) != T ||
      out.extent(1) != F)
    throw std::invalid_argument(
        "inkling_softcapped_swiglu: dimension mismatch");
  if (T == 0 || F == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, F}),
      KOKKOS_LAMBDA(size_t t, size_t f) {
        const float g = static_cast<float>(gate(t, f));
        const float u = static_cast<float>(up(t, f));
        const float gs = Kokkos::tanhf(g / gate_softcap) * gate_softcap;
        const float sig = 1.0f / (1.0f + Kokkos::expf(-g));
        const float us = Kokkos::tanhf(u / up_softcap) * up_softcap;
        out(t, f) = static_cast<Scalar>(gs * sig * us);
      });
}

// GatedOutputProjection elementwise gate multiply:
//   out[t, d] = attn_out[t, d] * sigmoid(gate[t, d])
// where gate is the caller-computed gate_proj(hidden) output and the
// caller performs the surrounding GEMMs. attn_out, gate, out: [T, D].
// In-place safe for attn_out/out.
template <class ExecSpace, class Scalar>
void inkling_gated_output_mul(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& attn_out,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::inkling_gated_output_mul") {
  const size_t T = attn_out.extent(0), D = attn_out.extent(1);
  if (gate.extent(0) != T || gate.extent(1) != D || out.extent(0) != T ||
      out.extent(1) != D)
    throw std::invalid_argument("inkling_gated_output_mul: dim mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        const float g = static_cast<float>(gate(t, d));
        const float sig = 1.0f / (1.0f + Kokkos::expf(-g));
        out(t, d) = static_cast<Scalar>(
            static_cast<float>(attn_out(t, d)) * sig);
      });
}

namespace Impl {
// Rank-64 message-gate hidden state: h = down([rms(x), rms(msg)]) with
// down: [2D, R] row-major, computed in FP32.
template <class ExecSpace, class Scalar>
void inkling_message_gate_hidden(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& msg,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& down, float eps,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& h) {
  const size_t T = x.extent(0), D = x.extent(1);
  const size_t R = down.extent(1);
  Kokkos::parallel_for(
      "KokkosDNN::inkling_message_gate_hidden",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, R}),
      KOKKOS_LAMBDA(size_t t, size_t r) {
        float ssx = 0, ssm = 0;
        for (size_t d = 0; d < D; ++d) {
          const float xv = static_cast<float>(x(t, d));
          const float mv = static_cast<float>(msg(t, d));
          ssx += xv * xv;
          ssm += mv * mv;
        }
        const float invx = 1.0f / Kokkos::sqrtf(ssx / static_cast<float>(D) + eps);
        const float invm = 1.0f / Kokkos::sqrtf(ssm / static_cast<float>(D) + eps);
        float acc = 0;
        for (size_t d = 0; d < D; ++d) {
          acc += static_cast<float>(x(t, d)) * invx * down(d, r);
          acc += static_cast<float>(msg(t, d)) * invm * down(D + d, r);
        }
        h(t, r) = acc;
      });
}
}  // namespace Impl

// MessageGate: out = (1 - tanh(forget*h)) * x + tanh(gain*h) * msg with
// h = down_proj([rms(x), rms(msg)]), all in FP32.
// x, msg, out: [T, D]; down: [2D, R] FP32; forget, gain: [R, D] FP32.
// Matches Inkling2LoopMessageGate. out must not alias x or msg (h is a
// separate workspace, but the final pass reads x/msg while writing out).
template <class ExecSpace, class Scalar>
void inkling_message_gate(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& msg,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& down,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& forget,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& gain,
    float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::inkling_message_gate") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (msg.extent(0) != T || msg.extent(1) != D || out.extent(0) != T ||
      out.extent(1) != D || down.extent(0) != 2 * D)
    throw std::invalid_argument("inkling_message_gate: dimension mismatch");
  const size_t R = down.extent(1);
  if (forget.extent(0) != R || forget.extent(1) != D || gain.extent(0) != R ||
      gain.extent(1) != D)
    throw std::invalid_argument(
        "inkling_message_gate: gate dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> h("inkling_msg_h", T,
                                                          R);
  Impl::inkling_message_gate_hidden<ExecSpace, Scalar>(x, msg, down, eps, h);
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float f = 0, g = 0;
        for (size_t r = 0; r < R; ++r) {
          f += forget(r, d) * h(t, r);
          g += gain(r, d) * h(t, r);
        }
        const float fgate = Kokkos::tanhf(f);
        const float mgate = Kokkos::tanhf(g);
        out(t, d) = static_cast<Scalar>(
            (1.0f - fgate) * static_cast<float>(x(t, d)) +
            mgate * static_cast<float>(msg(t, d)));
      });
}

}  // namespace KokkosDNN

#endif
