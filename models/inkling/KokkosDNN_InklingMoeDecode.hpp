// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGMOEDECODE_HPP
#define KOKKOSDNN_INKLINGMOEDECODE_HPP

// Fused MoE decode for the Inkling 2 Mini text decoder (one token).
//
// The legacy path (Model::moe_forward) routes on the host: per token, per
// layer visit, 2 D2H fences + a 128-expert host scan + 2 H2D fences per
// routed expert, then gather + gate_up GEMV + 2 split copies + swiglu + down
// GEMV + scatter per expert (~8 launches + fences per expert). The fused
// path does it in three launches per visit with zero host fences:
//
//   0. quantize the shared input row (and the S slot rows for stage 2)
//   1. gate_up int8 dots + softcapped SwiGLU -> act [S, I]
//   2. down int8 dots + weighted slot combine  -> out [1, D] (written)
//
// Routing idx/wts are consumed on device (moe_route_sigmoid already leaves
// them there); at C == 1 every slot is its own row, so no bucketing is
// needed. Dots reuse the portable int8 machinery (GgufMvq unit windows and
// integer dots: same values as quant_gemv's int8 path up to FP
// reassociation). Weights stay packed: no dequant staging, no dense traffic.

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_GGUFGemvInt8.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Occupancy.hpp"

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

namespace Mvq = ::KokkosDNN::Impl::GgufMvq;

// True when the fused decode path can serve these packed expert shapes: the
// int8-dot path must cover the dtype (IQ4_NL, Q8_0) at these widths.
inline bool moe_decode_fused_supported(Gguf::GgmlType gu_dtype, size_t K,
                                       Gguf::GgmlType down_dtype, size_t I) {
  Mvq::Wd gu_wd{}, down_wd{};
  if (!Gguf::Impl::quant_int8_dtype(gu_dtype, gu_wd)) return false;
  if (!Gguf::Impl::quant_int8_dtype(down_dtype, down_wd)) return false;
  return Mvq::dtype_ok(gu_wd, K) && Mvq::dtype_ok(down_wd, I);
}

// Stage 1: act[s, j] = softcap(g_e.j . x) * sigmoid(g_e.j . x) *
// softcap(u_e.j . x) with the fused gate_up rows (gate j, up j+I) of expert
// e = idx[s]. One team per (slot, output): threads reduce over the K/32
// int8 units with integer dots against the shared quantized input window,
// then a single thread applies the activation.
template <class ExecSpace, class Scalar, Mvq::Wd DT>
struct InklingMoeGateUpFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using ActView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using IdxView = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>;
  const int8_t* xq;
  const Mvq::ActQ8Hdr* xh;
  Kokkos::View<uintptr_t*, ExecSpace> gu_base;
  Gguf::GgmlType gu_dtype;
  size_t gu_row_bytes;
  IdxView idx;  // [1, S] expert per slot
  ActView act;  // [S, I]
  size_t I, nunits;
  float gate_softcap, up_softcap;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const ActView act_ = act;
    const IdxView idx_ = idx;
    const int8_t* xq_ = xq;
    const Mvq::ActQ8Hdr* xh_ = xh;
    const size_t I_ = I, nu_ = nunits;
    const float gsc_ = gate_softcap, usc_ = up_softcap;
    const size_t rb_ = gu_row_bytes;
    const size_t s = static_cast<size_t>(team.league_rank()) / I_;
    const size_t j = static_cast<size_t>(team.league_rank()) % I_;
    const size_t e = static_cast<size_t>(idx_(0, s));
    const uint8_t* payload =
        reinterpret_cast<const uint8_t*>(gu_base(e));
    const uint8_t* rg = payload + j * rb_;
    const uint8_t* ru = payload + (j + I_) * rb_;
    float g = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nu_),
        KOKKOS_LAMBDA(size_t c, float& acc) {
          const Mvq::Int8UnitWindow xw =
              Mvq::mvq_load_unit_window<ExecSpace, DT>(xq_, xh_, (int)c);
          acc += Mvq::gemv_int8_unit_dot_win<ExecSpace, DT>(rg, (int)c, xw);
        },
        g);
    float u = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nu_),
        KOKKOS_LAMBDA(size_t c, float& acc) {
          const Mvq::Int8UnitWindow xw =
              Mvq::mvq_load_unit_window<ExecSpace, DT>(xq_, xh_, (int)c);
          acc += Mvq::gemv_int8_unit_dot_win<ExecSpace, DT>(ru, (int)c, xw);
        },
        u);
    // Integer dots land fully scaled (weight_scale x act_scale fold into
    // the unit dot, as in quant_gemv's int8 path); the single thread only
    // applies the softcapped activation.
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      const float gs = Kokkos::tanhf(g / gsc_) * gsc_;
      const float sig = 1.0f / (1.0f + Kokkos::expf(-g));
      const float us = Kokkos::tanhf(u / usc_) * usc_;
      act_(s, j) = static_cast<Scalar>(gs * sig * us);
    });
  }
};

// Stage 2: out[d] = sum_s wts[s] * (down_e[s][d, :] . act[s, :]) with int8
// dots against the per-slot quantized activation rows. One team per output
// dim; slots accumulate serially (S <= 8), units in parallel.
template <class ExecSpace, class Scalar, Mvq::Wd DT>
struct InklingMoeDownFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using OutView = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using IdxView = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>;
  using WtsView = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  const int8_t* aq;
  const Mvq::ActQ8Hdr* ah;
  Kokkos::View<uintptr_t*, ExecSpace> down_base;
  Gguf::GgmlType down_dtype;
  size_t down_row_bytes;
  IdxView idx;  // [1, S]
  WtsView wts;  // [1, S]
  OutView out;  // [1, D], written
  size_t I, S, nunits;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const IdxView idx_ = idx;
    const WtsView wts_ = wts;
    const OutView out_ = out;
    const int8_t* aq_ = aq;
    const Mvq::ActQ8Hdr* ah_ = ah;
    const size_t I_ = I, S_ = S, nu_ = nunits;
    const size_t rb_ = down_row_bytes;
    const size_t d = static_cast<size_t>(team.league_rank());
    float acc = 0;
    for (size_t s = 0; s < S_; ++s) {
      const size_t e = static_cast<size_t>(idx_(0, s));
      const float w = wts_(0, s);
      const uint8_t* row =
          reinterpret_cast<const uint8_t*>(down_base(e)) + d * rb_;
      float r = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, nu_),
          KOKKOS_LAMBDA(size_t c, float& sum) {
            const Mvq::Int8UnitWindow aw =
                Mvq::mvq_load_unit_window<ExecSpace, DT>(
                    aq_ + s * I_, ah_ + s * (I_ / 32), (int)c);
            sum += Mvq::gemv_int8_unit_dot_win<ExecSpace, DT>(row, (int)c, aw);
          },
          r);
      acc += w * r;
    }
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      out_(0, d) = static_cast<Scalar>(acc);
    });
  }
};

// Three-launch fused MoE decode for one token. See the header docs for the
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
    const Kokkos::View<uint8_t*, ExecSpace>& qx_scratch,
    const Kokkos::View<uint8_t*, ExecSpace>& qa_scratch,
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
  Mvq::Wd gu_wd{}, down_wd{};
  if (!Gguf::Impl::quant_int8_dtype(gu.dtype, gu_wd) ||
      !Gguf::Impl::quant_int8_dtype(down.dtype, down_wd))
    throw std::invalid_argument(
        "inkling_moe_decode_fused: expert dtype lacks an int8 path");
  if (!Mvq::dtype_ok(gu_wd, K) || !Mvq::dtype_ok(down_wd, I))
    throw std::invalid_argument(
        "inkling_moe_decode_fused: expert width not int8 aligned");
  const size_t xneed = Mvq::act_q8_bytes(K);
  const size_t aneed = Mvq::act_q8_bytes(S * I);
  if (qx_scratch.extent(0) < xneed || qa_scratch.extent(0) < aneed)
    throw std::invalid_argument(
        "inkling_moe_decode_fused: activation scratch too small");
  // Stage 0: quantize the shared input row.
  {
    uint8_t* const sbase = qx_scratch.data();
    int8_t* const xq = reinterpret_cast<int8_t*>(sbase);
    Mvq::ActQ8Hdr* const xh = reinterpret_cast<Mvq::ActQ8Hdr*>(
        sbase + Mvq::act_q8_hdr_offset(K));
    const int nblk = (int)Mvq::act_q8_blocks(K);
    Mvq::QuantizeActQ8Functor<ExecSpace, Scalar> qf{x.data(), xq, xh, nblk};
    Kokkos::parallel_for(label + "::quant_x",
                          Kokkos::RangePolicy<ExecSpace>(0, nblk), qf);
  }
  // Stage 1: gate_up int8 dots + swiglu.
  const auto launch_gate_up = [&](auto wd_tag) {
    constexpr Mvq::Wd DT = decltype(wd_tag)::value;
    InklingMoeGateUpFunctor<ExecSpace, Scalar, DT> f{
        reinterpret_cast<const int8_t*>(qx_scratch.data()),
        reinterpret_cast<const Mvq::ActQ8Hdr*>(qx_scratch.data() +
                                               Mvq::act_q8_hdr_offset(K)),
        gu.base, gu.dtype, gu.row_bytes, idx, act, I, K / 32,
        gate_softcap, up_softcap};
    // Clamp to the backend maximum (Serial -> 1); the strided
    // TeamThreadRange stays correct at any team size.
    int gate_up_ts =
        ::KokkosDNN::Impl::cached_team_size_max<ExecSpace>(f);
    if (gate_up_ts > 128) gate_up_ts = 128;
    if (gate_up_ts < 1) gate_up_ts = 1;
    Kokkos::TeamPolicy<ExecSpace> policy((int)(S * I), gate_up_ts);
    Kokkos::parallel_for(label + "::gate_up", policy, f);
  };
  switch (gu_wd) {
    case Mvq::Wd::IQ4NL:
      launch_gate_up(std::integral_constant<Mvq::Wd, Mvq::Wd::IQ4NL>{});
      break;
    case Mvq::Wd::Q80:
      launch_gate_up(std::integral_constant<Mvq::Wd, Mvq::Wd::Q80>{});
      break;
    default:
      throw std::invalid_argument(
          "inkling_moe_decode_fused: unsupported gate_up dtype");
  }
  // Stage 0b: quantize the slot rows (flattened [S*I]). Same-stream
  // ordering covers the stage-1 -> quantize dependency (no fence).
  {
    uint8_t* const sbase = qa_scratch.data();
    int8_t* const aq = reinterpret_cast<int8_t*>(sbase);
    Mvq::ActQ8Hdr* const ah = reinterpret_cast<Mvq::ActQ8Hdr*>(
        sbase + Mvq::act_q8_hdr_offset(S * I));
    const int nblk = (int)Mvq::act_q8_blocks(S * I);
    Mvq::QuantizeActQ8Functor<ExecSpace, Scalar> qf{act.data(), aq, ah, nblk};
    Kokkos::parallel_for(label + "::quant_a",
                          Kokkos::RangePolicy<ExecSpace>(0, nblk), qf);
  }
  // Stage 2: down int8 dots + weighted combine.
  const auto launch_down = [&](auto wd_tag) {
    constexpr Mvq::Wd DT = decltype(wd_tag)::value;
    InklingMoeDownFunctor<ExecSpace, Scalar, DT> f{
        reinterpret_cast<const int8_t*>(qa_scratch.data()),
        reinterpret_cast<const Mvq::ActQ8Hdr*>(qa_scratch.data() +
                                               Mvq::act_q8_hdr_offset(S * I)),
        down.base, down.dtype, down.row_bytes, idx, wts, out, I, S, I / 32};
    // Clamp to the backend maximum (Serial -> 1); the strided
    // TeamThreadRange stays correct at any team size.
    int down_ts = ::KokkosDNN::Impl::cached_team_size_max<ExecSpace>(f);
    if (down_ts > 128) down_ts = 128;
    if (down_ts < 1) down_ts = 1;
    Kokkos::TeamPolicy<ExecSpace> policy((int)D, down_ts);
    Kokkos::parallel_for(label + "::down", policy, f);
  };
  switch (down_wd) {
    case Mvq::Wd::IQ4NL:
      launch_down(std::integral_constant<Mvq::Wd, Mvq::Wd::IQ4NL>{});
      break;
    case Mvq::Wd::Q80:
      launch_down(std::integral_constant<Mvq::Wd, Mvq::Wd::Q80>{});
      break;
    default:
      throw std::invalid_argument(
          "inkling_moe_decode_fused: unsupported down dtype");
  }
}

}  // namespace Impl
}  // namespace Inkling
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_INKLINGMOEDECODE_HPP
