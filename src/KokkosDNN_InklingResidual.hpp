// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGRESIDUAL_HPP
#define KOKKOSDNN_INKLINGRESIDUAL_HPP

// Inkling2 loop attention-residual retrievals
// (thinkingmachines/Inkling-2-Mini text decoder): BlockAttentionResidual (1
// or 3 learned queries) and MessageAttentionResidual (per-loop-layer query
// table producing 4 messages). Per token: unweighted-RMSNorm the stacked
// [block_states; partial] rows, then single-head softmax attention with
// scale 1/sqrt(D), values = normalized states. Batch = 1 with T-major
// layouts. Plain Kokkos (no cudnn); all accumulation in FP32 with the
// result cast back to Scalar. Backend-portable: only parallel_for kernels;
// no backend-specific code.

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Impl {

// One (token, query) worker: norms the S+1 stacked rows (block history +
// partial), attends with the Q-th query row, writes one [D] output row.
// Team-parallel over the model dim: one team per (t, q), threads split D,
// with the S+1 norms/scores in team scratch. The legacy serial form left a
// decode row to one thread grinding ~200KB through dependent chains
// (~0.7 ms/launch on RTX 3090); the team form is ~50x faster. Reduction
// order across the team's threads differs from the serial loop, so results
// agree to FP reassociation (~1e-7 relative), not bit-for-bit.
template <class ExecSpace, class Scalar, class BView, class PView, class QView,
          class OView>
struct InklingAttnResidualWorker {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  BView blocks;  // [T, S, D]
  PView partial;  // [T, D]
  QView query;    // [Q, D] FP32
  OView out;      // [T, Q, D]
  size_t S;
  size_t Q;
  size_t D;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    // Cap from the callers (S <= 64); scratch holds one norm + one score
    // per stacked row.
    constexpr size_t kCap = 65;
    const size_t S_ = S, Q_ = Q, D_ = D;
    const float eps_ = eps;
    const BView blocks_ = blocks;
    const PView partial_ = partial;
    const QView query_ = query;
    const OView out_ = out;
    const size_t t = static_cast<size_t>(team.league_rank()) / Q_;
    const size_t q = static_cast<size_t>(team.league_rank()) % Q_;
    const float scale = 1.0f / Kokkos::sqrtf(static_cast<float>(D_));
    float* buf = static_cast<float*>(
        team.team_scratch(0).get_shmem(sizeof(float) * (2 * kCap + 2)));
    float* inv = buf;
    float* scr = buf + kCap;
    float* m_den = buf + 2 * kCap;  // [0] = max, [1] = denom (scratch so
                                    // [=]-captured singles can update them)
    // Norms of the stacked rows (one TeamThreadRange reduce per row).
    for (size_t s = 0; s <= S_; ++s) {
      float ss = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, D_),
          KOKKOS_LAMBDA(size_t d, float& a) {
            const float v = (s < S_) ? static_cast<float>(blocks_(t, s, d))
                                     : static_cast<float>(partial_(t, d));
            a += v * v;
          },
          ss);
      Kokkos::single(
          Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
            inv[s] = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D_) + eps_);
          });
    }
    team.team_barrier();
    // Scores + softmax over the stacked rows (serial over S+1 <= 65 rows).
    Kokkos::single(
        Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
          float m = -1e30f;
          for (size_t s = 0; s <= S_; ++s) {
            float a = 0;
            for (size_t d = 0; d < D_; ++d) {
              const float v = (s < S_) ? static_cast<float>(blocks_(t, s, d))
                                       : static_cast<float>(partial_(t, d));
              a += static_cast<float>(query_(q, d)) * v * inv[s];
            }
            scr[s] = a * scale;
            m = (scr[s] > m) ? scr[s] : m;
          }
          float denom = 0;
          for (size_t s = 0; s <= S_; ++s) {
            scr[s] = Kokkos::expf(scr[s] - m);
            denom += scr[s];
          }
          m_den[0] = m;
          m_den[1] = denom;
        });
    team.team_barrier();
    const float rden = 1.0f / m_den[1];
    // Weighted sum, threads split d, rows ascending like the serial form.
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          float acc = 0;
          for (size_t s = 0; s <= S_; ++s) {
            const float v = (s < S_) ? static_cast<float>(blocks_(t, s, d))
                                     : static_cast<float>(partial_(t, d));
            acc += scr[s] * rden * v * inv[s];
          }
          out_(t, q, d) = static_cast<Scalar>(acc);
        });
    (void)Q_;
  }
};

}  // namespace Impl

// BlockAttentionResidual: Q learned queries (Q = 1, or 3 for
// full-attention loop layers) shared across positions retrieve from the
// per-token stack [block_states[t]; partial[t]].
// block_states: [T, S, D] block history (S blocks so far, S may be 0);
// partial: [T, D] current hidden; query: [Q, D] learned queries (FP32 or
// model Scalar); out: [T, Q, D]. Matches Inkling2LoopBlockAttentionResidual
// (single-head attention, scale hidden^-0.5, values = normalized states).
template <class ExecSpace, class Scalar, class QView>
void inkling_block_attention_residual(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& block_states,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& partial,
    const QView& query,
    float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::inkling_block_attention_residual") {
  const size_t T = partial.extent(0), D = partial.extent(1);
  const size_t S = block_states.extent(1);
  const size_t Q = query.extent(0);
  if (block_states.extent(0) != T || block_states.extent(2) != D ||
      query.extent(1) != D || out.extent(0) != T || out.extent(1) != Q ||
      out.extent(2) != D)
    throw std::invalid_argument(
        "inkling_block_attention_residual: dimension mismatch");
  if (S > 64)
    throw std::invalid_argument(
        "inkling_block_attention_residual: S exceeds worker capacity (64)");
  if (T == 0 || D == 0 || Q == 0) return;
  Impl::InklingAttnResidualWorker<
      ExecSpace, Scalar,
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>,
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>, QView,
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>
      w{block_states, partial, query, out, S, Q, D, eps};
  // One team per (t, q); threads split D (see the worker). Scratch holds
  // 2*65 norms/scores plus max/denom.
  int team_size = 128;
  if ((int)D < team_size) {
    team_size = 1;
    while (team_size < (int)D) team_size *= 2;
  }
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  Policy policy((int)(T * Q), team_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(float) * (2 * 65 + 2)));
  Kokkos::parallel_for(label, policy, w);
}

// MessageAttentionResidual: per-loop-layer query table [L, 4, D] (FP32)
// retrieves the 4 messages (q, k, v, mlp inputs) from the previous pass's
// per-token stack [block_states[t]; partial[t]].
// block_states: [T, S, D] previous-pass block states; partial: [T, D]
// previous-pass hidden; query_table: [L, 4, D]; loop_layer_idx selects the
// layer row; out: [T, 4, D]. Matches Inkling2LoopMessageAttentionResidual.
template <class ExecSpace, class Scalar>
void inkling_message_attention_residual(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& block_states,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& partial,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& query_table,
    size_t loop_layer_idx, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label =
        "KokkosDNN::inkling_message_attention_residual") {
  const size_t T = partial.extent(0), D = partial.extent(1);
  const size_t S = block_states.extent(1);
  if (block_states.extent(0) != T || block_states.extent(2) != D ||
      query_table.extent(1) != 4 || query_table.extent(2) != D ||
      loop_layer_idx >= query_table.extent(0) || out.extent(0) != T ||
      out.extent(1) != 4 || out.extent(2) != D)
    throw std::invalid_argument(
        "inkling_message_attention_residual: dimension mismatch");
  if (S > 64)
    throw std::invalid_argument(
        "inkling_message_attention_residual: S exceeds worker capacity (64)");
  if (T == 0 || D == 0) return;
  // Row subview of the table: unmanaged [4, D] alias, no copy. Offset
  // arithmetic (not &table(..)) so this also works for device views.
  using QV = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  QV query(query_table.data() +
               (loop_layer_idx * 4 + size_t(0)) * D,
           4, D);
  Impl::InklingAttnResidualWorker<
      ExecSpace, Scalar,
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>,
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>, QV,
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>
      w{block_states, partial, query, out, S, 4, D, eps};
  // One team per (t, q); threads split D (see the worker). Scratch holds
  // 2*65 norms/scores plus max/denom.
  int team_size = 128;
  if ((int)D < team_size) {
    team_size = 1;
    while (team_size < (int)D) team_size *= 2;
  }
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  Policy policy((int)(T * 4), team_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(float) * (2 * 65 + 2)));
  Kokkos::parallel_for(label, policy, w);
}

}  // namespace KokkosDNN

#endif
