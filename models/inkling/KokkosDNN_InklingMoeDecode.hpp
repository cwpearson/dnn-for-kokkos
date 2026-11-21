// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGMOEDECODE_HPP
#define KOKKOSDNN_INKLINGMOEDECODE_HPP

// Fused MoE decode for the Inkling 2 Mini text decoder (one token).
//
// The legacy path (Model::moe_forward) routes on the host: per token it
// deep_copies idx/wts to host, loops 128 experts, and launches gather +
// gate_up GEMV + 2 split copies + swiglu + down GEMV + scatter per routed
// expert (~8 launches + 2 fences per expert, ~350us/visit on RTX 3090).
// This path does it in two launches per layer visit with zero host fences:
//
//   1. gate_up fused dots + softcapped SwiGLU -> act [S, I]
//   2. down fused dots + weighted slot combine  -> out [1, D] (written)
//
// Routing idx/wts are consumed on device (moe_route_sigmoid already leaves
// them there); at C == 1 every slot is its own row, so no host-side bucket
// grouping is needed at all. Dots reuse the tested grouped primitives
// (Gguf::quant_group_dot, same values as quant_gemm's grouped path up to FP
// reassociation). Weights stay packed: no dequant staging, no dense traffic.

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Inkling {
namespace Impl {

// Host-side cache of device expert tables: make_quant_weight_table copies
// 128 pointers H2D, so rebuilding per visit would cost a fence per layer.
// Weights are immutable during forward(), so rebuild only when the source
// vectors change (never in steady state).
template <class ExecSpace>
struct InklingMoeTableCache {
  Gguf::QuantWeightTable<ExecSpace> gu, down;
  const void* gu_vec = nullptr;
  const void* down_vec = nullptr;
  size_t gu_n = 0, down_n = 0;
  const void* gu0 = nullptr;
  const void* down0 = nullptr;

  void ensure(const std::vector<Gguf::QuantWeight<ExecSpace>>& gu_ws,
              const std::vector<Gguf::QuantWeight<ExecSpace>>& down_ws) {
    const void* n_gu_vec = static_cast<const void*>(gu_ws.data());
    const void* n_down_vec = static_cast<const void*>(down_ws.data());
    const void* n_gu0 = gu_ws.empty() ? nullptr : gu_ws[0].data.data();
    const void* n_down0 =
        down_ws.empty() ? nullptr : down_ws[0].data.data();
    if (n_gu_vec == gu_vec && n_down_vec == down_vec && gu_ws.size() == gu_n &&
        down_ws.size() == down_n && n_gu0 == gu0 && n_down0 == down0)
      return;
    gu = Gguf::make_quant_weight_table<ExecSpace>(gu_ws,
                                                  "Inkling::moe_gu_table");
    down = Gguf::make_quant_weight_table<ExecSpace>(
        down_ws, "Inkling::moe_down_table");
    gu_vec = n_gu_vec;
    down_vec = n_down_vec;
    gu_n = gu_ws.size();
    down_n = down_ws.size();
    gu0 = n_gu0;
    down0 = n_down0;
  }
};

// True when the fused decode path can serve these packed expert shapes:
// both projections must take the grouped dot path (one scale set per 32
// elements). IQ4_NL and Q8_0 qualify at the Inkling widths (K = 2304,
// I = 1152); K-quants do not divide evenly and keep the legacy path.
inline bool moe_decode_fused_supported(Gguf::GgmlType gu_dtype, size_t K,
                                       Gguf::GgmlType down_dtype, size_t I) {
  size_t gs = 0;
  if (!Gguf::Impl::quant_can_group(gu_dtype, K, gs)) return false;
  if (!Gguf::Impl::quant_can_group(down_dtype, I, gs)) return false;
  return true;
}

// Stage 1: act[s, j] = softcap(g_e.j . x) * sigmoid(g_e.j . x) *
// softcap(u_e.j . x) with the fused gate_up rows (gate j, up j+I) of expert
// e = idx[s]. One team per (slot, output): threads reduce over the K/32
// groups, gate then up, then a single thread applies the activation.
template <class ExecSpace, class Scalar>
struct InklingMoeGateUpFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using XView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using ActView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using IdxView = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>;
  XView x;  // [1, K], single token row
  Kokkos::View<uintptr_t*, ExecSpace> gu_base;
  Gguf::GgmlType gu_dtype;
  size_t gu_row_bytes;
  IdxView idx;  // [1, S] expert per slot
  ActView act;  // [S, I]
  size_t I, ngroups, gs;
  float gate_softcap, up_softcap;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const XView x_ = x;
    const ActView act_ = act;
    const IdxView idx_ = idx;
    const auto gu_base_ = gu_base;
    const size_t I_ = I, ng_ = ngroups, gs_ = gs;
    const float gsc_ = gate_softcap, usc_ = up_softcap;
    const Gguf::GgmlType dt_ = gu_dtype;
    const size_t rb_ = gu_row_bytes;
    const size_t s = static_cast<size_t>(team.league_rank()) / I_;
    const size_t j = static_cast<size_t>(team.league_rank()) % I_;
    const size_t e = static_cast<size_t>(idx_(0, s));
    const uint8_t* payload =
        reinterpret_cast<const uint8_t*>(gu_base_(e));
    const uint8_t* rg = payload + j * rb_;
    const uint8_t* ru = payload + (j + I_) * rb_;
    const Scalar* xd = x_.data();
    const size_t xs = x_.stride(1);
    float g = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, ng_),
        KOKKOS_LAMBDA(size_t gg, float& acc) {
          acc += Gguf::Impl::quant_group_dot(dt_, rg, xd, xs, gg * gs_);
        },
        g);
    float u = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, ng_),
        KOKKOS_LAMBDA(size_t gg, float& acc) {
          acc += Gguf::Impl::quant_group_dot(dt_, ru, xd, xs, gg * gs_);
        },
        u);
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      const float gs = Kokkos::tanhf(g / gsc_) * gsc_;
      const float sig = 1.0f / (1.0f + Kokkos::expf(-g));
      const float us = Kokkos::tanhf(u / usc_) * usc_;
      act_(s, j) = static_cast<Scalar>(gs * sig * us);
    });
  }
};

// Stage 2: out[d] = sum_s wts[s] * (down_e[s][d, :] . act[s, :]). One team
// per output dim; slots accumulate serially (S <= 8), groups in parallel.
template <class ExecSpace, class Scalar>
struct InklingMoeDownFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using ActView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using IdxView = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>;
  using WtsView = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using OutView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  ActView act;  // [S, I]
  Kokkos::View<uintptr_t*, ExecSpace> down_base;
  Gguf::GgmlType down_dtype;
  size_t down_row_bytes;
  IdxView idx;  // [1, S]
  WtsView wts;  // [1, S]
  OutView out;  // [1, D], written
  size_t I, S, ngroups, gs;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const ActView act_ = act;
    const IdxView idx_ = idx;
    const WtsView wts_ = wts;
    const OutView out_ = out;
    const size_t I_ = I, S_ = S, ng_ = ngroups, gs_ = gs;
    const Gguf::GgmlType dt_ = down_dtype;
    const size_t rb_ = down_row_bytes;
    const size_t d = static_cast<size_t>(team.league_rank());
    float acc = 0;
    for (size_t s = 0; s < S_; ++s) {
      const size_t e = static_cast<size_t>(idx_(0, s));
      const float w = wts_(0, s);
      const uint8_t* row =
          reinterpret_cast<const uint8_t*>(down_base(e)) + d * rb_;
      const Scalar* ad = act_.data() + s * act_.stride(0);
      const size_t axs = act_.stride(1);
      float r = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, ng_),
          KOKKOS_LAMBDA(size_t gg, float& sum) {
            sum += Gguf::Impl::quant_group_dot(dt_, row, ad, axs, gg * gs_);
          },
          r);
      acc += w * r;
    }
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      out_(0, d) = static_cast<Scalar>(acc);
    });
  }
};

// Two-launch fused MoE decode for one token. See the header docs for the
// contract. Throws on shape/dtype misuse; the caller checks
// moe_decode_fused_supported() first to choose this path vs the legacy
// host-grouped one.
template <class ExecSpace, class Scalar>
void inkling_moe_decode_fused(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Gguf::QuantWeightTable<ExecSpace>& gu,
    const Gguf::QuantWeightTable<ExecSpace>& down,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& wts,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& act,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    float gate_softcap, float up_softcap,
    const std::string& label = "KokkosDNN::Inkling::moe_decode_fused") {
  const size_t S = idx.extent(1);
  if (x.extent(0) != 1 || out.extent(0) != 1)
    throw std::invalid_argument(
        "inkling_moe_decode_fused: single-token decode only");
  if (idx.extent(0) != 1 || wts.extent(0) != 1 || wts.extent(1) != S)
    throw std::invalid_argument(
        "inkling_moe_decode_fused: routing dimension mismatch");
  const size_t K = x.extent(1), D = out.extent(1);
  const size_t I = act.extent(1);
  if (act.extent(0) != S || gu.cols != K || gu.rows != 2 * I ||
      down.cols != I || down.rows != D)
    throw std::invalid_argument(
        "inkling_moe_decode_fused: dimension mismatch");
  if (S == 0 || K == 0 || D == 0 || I == 0) return;
  size_t gs_gu = 0, gs_d = 0;
  if (!Gguf::Impl::quant_can_group(gu.dtype, K, gs_gu) ||
      !Gguf::Impl::quant_can_group(down.dtype, I, gs_d))
    throw std::invalid_argument(
        "inkling_moe_decode_fused: expert dtype not grouped-dot eligible");
  {
    InklingMoeGateUpFunctor<ExecSpace, Scalar> f{
        x,          gu.base, gu.dtype, gu.row_bytes, idx, act,
        I,          K / gs_gu, gs_gu, gate_softcap, up_softcap};
    Kokkos::TeamPolicy<ExecSpace> policy((int)(S * I), 128);
    Kokkos::parallel_for(label + "::gate_up", policy, f);
  }
  {
    InklingMoeDownFunctor<ExecSpace, Scalar> f{
        act, down.base, down.dtype, down.row_bytes, idx, wts,
        out, I, S, I / gs_d, gs_d};
    Kokkos::TeamPolicy<ExecSpace> policy((int)D, 128);
    Kokkos::parallel_for(label + "::down", policy, f);
  }
}

}  // namespace Impl
}  // namespace Inkling
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_INKLINGMOEDECODE_HPP
