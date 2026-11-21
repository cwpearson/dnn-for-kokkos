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

#include "KokkosDNN_RMSNorm.hpp"

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
// Team-parallel message-gate hidden state: h = down([rms(x), rms(msg)]) with
// down: [2D, R] row-major, computed in FP32. One team per (t, r), threads
// reduce over the 2D concat (logically indexed, never materialized). The
// legacy serial form left one thread per (t, r) grinding 2x2304-element dots
// through dependent chains (~176 us/launch on RTX 3090); the team form is
// ~20x faster. Reduction order differs from the serial loop, so results
// agree to FP reassociation, not bit-for-bit.
template <class ExecSpace, class Scalar, class DView>
struct MsgGateHiddenTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using RmsView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using HView = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  RmsView rms_x;
  RmsView rms_m;
  DView down;
  HView h;
  size_t T, D, R;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const RmsView rms_x_ = rms_x;
    const RmsView rms_m_ = rms_m;
    const DView down_ = down;
    const HView h_ = h;
    const size_t D_ = D, R_ = R;
    const size_t t = static_cast<size_t>(team.league_rank()) / R_;
    const size_t r = static_cast<size_t>(team.league_rank()) % R_;
    float acc = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, 2 * D_),
        KOKKOS_LAMBDA(size_t d, float& a) {
          const float v = (d < D_) ? static_cast<float>(rms_x_(t, d))
                                   : static_cast<float>(rms_m_(t, d - D_));
          a += v * static_cast<float>(down_(d, r));
        },
        acc);
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() { h_(t, r) = acc; });
  }
};

// Rank-64 message-gate hidden state from PRE-NORMED streams (see
// inkling_message_gate, which norms first): h = down([rms_x, rms_m]).
template <class ExecSpace, class Scalar, class DView>
void inkling_message_gate_hidden(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& rms_x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& rms_m,
    const DView& down,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& h,
    const std::string& label = "KokkosDNN::inkling_message_gate_hidden") {
  const size_t T = rms_x.extent(0), D = rms_x.extent(1);
  const size_t R = down.extent(1);
  if (rms_m.extent(0) != T || rms_m.extent(1) != D ||
      down.extent(0) != 2 * D || h.extent(0) != T || h.extent(1) != R)
    throw std::invalid_argument(
        "inkling_message_gate_hidden: dimension mismatch");
  if (T == 0 || D == 0 || R == 0) return;
  int team_size = 128;
  if ((int)(2 * D) < team_size) {
    team_size = 1;
    while (team_size < (int)(2 * D)) team_size *= 2;
  }
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  Policy policy((int)(T * R), team_size);
  MsgGateHiddenTeam<ExecSpace, Scalar, DView> w{rms_x, rms_m, down, h, T, D,
                                                R};
  Kokkos::parallel_for(label, policy, w);
}
}  // namespace Impl

// MessageGate: out = (1 - tanh(forget*h)) * x + tanh(gain*h) * msg with
// h = down_proj([rms(x), rms(msg)]), all in FP32.
// x, msg, out: [T, D]; down: [2D, R] FP32; forget, gain: [R, D] FP32.
// Matches Inkling2LoopMessageGate. out must not alias x or msg (the final
// pass reads x/msg while writing out). rms_x, rms_m ([T, D]) and h ([T, R])
// are caller-owned scratch: the norms and hidden state are too big to
// recompute per call at decode, and allocating them per call costs a device
// allocation per gate (4+ per layer visit).
template <class ExecSpace, class Scalar, class DView, class FView,
          class GView>
void inkling_message_gate(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& msg,
    const DView& down,
    const FView& forget,
    const GView& gain,
    float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& rms_x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& rms_m,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& h,
    const std::string& label = "KokkosDNN::inkling_message_gate") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (msg.extent(0) != T || msg.extent(1) != D || out.extent(0) != T ||
      out.extent(1) != D || down.extent(0) != 2 * D)
    throw std::invalid_argument("inkling_message_gate: dimension mismatch");
  const size_t R = down.extent(1);
  if (forget.extent(0) != R || forget.extent(1) != D || gain.extent(0) != R ||
      gain.extent(1) != D || rms_x.extent(0) != T || rms_x.extent(1) != D ||
      rms_m.extent(0) != T || rms_m.extent(1) != D || h.extent(0) != T ||
      h.extent(1) != R)
    throw std::invalid_argument(
        "inkling_message_gate: gate/scratch dimension mismatch");
  if (T == 0 || D == 0) return;
  rmsnorm_noscale<ExecSpace, Scalar>(x, eps, rms_x);
  rmsnorm_noscale<ExecSpace, Scalar>(msg, eps, rms_m);
  Impl::inkling_message_gate_hidden<ExecSpace, Scalar>(rms_x, rms_m, down, h);
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float f = 0, g = 0;
        for (size_t r = 0; r < R; ++r) {
          f += static_cast<float>(forget(r, d)) * h(t, r);
          g += static_cast<float>(gain(r, d)) * h(t, r);
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
