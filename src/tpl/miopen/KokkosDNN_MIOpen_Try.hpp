#ifndef KOKKOSDNN_MIOPEN_TRY_HPP
#define KOKKOSDNN_MIOPEN_TRY_HPP

// Layout-agnostic MIOpen SDPA dispatch over Kokkos views. Performs the
// cheap shape checks, maps the scalar type, and forwards raw
// pointers/strides along with the handle's cached MIOpen handle (created
// once per DNNHandle).
//
// STUB STATUS (see tpl/miopen/KokkosDNN_MIOpen_Forward.hpp): MIOpen
// exposes no fused SDPA/FlashAttention API, so the forward entry points
// always return false and every op runs the native kernel
// (native-by-construction). The checks below mirror try_cudnn_sdpa_*
// (float + bhalf_t only, Hq % Hkv == 0, causal Ktot == q_off + T,
// null-data guards) so a future hipBLAS + MIOpen-softmax composition path
// can drop in behind the same gate. Unlike the cuDNN path there is no
// D-contiguity repack here: no TPL kernel consumes the pointers yet, and
// the native kernel is layout-agnostic, so repacking would only burn a
// copy before falling back anyway.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <type_traits>

#include "KokkosDNN_Handle.hpp"
#include "tpl/miopen/KokkosDNN_MIOpen_Forward.hpp"

namespace KokkosDNN {
namespace Impl {

template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_miopen_sdpa_gqa_impl(const DNNHandle<Exec>& handle,
                                     const QView& q, const KView& k,
                                     const VView& v, size_t q_off,
                                     const OView& out, float scale,
                                     int64_t window) {
  // window < 0 selects full causal attention; window > 0 selects
  // sliding-window causal attention with that band width. window == 0 is
  // rejected (the sliding entry point below never forwards it; the causal
  // entry point passes -1).
  if (window == 0) return false;
  const int64_t T    = static_cast<int64_t>(q.extent(0));
  const int64_t Hq   = static_cast<int64_t>(q.extent(1));
  const int64_t D    = static_cast<int64_t>(q.extent(2));
  const int64_t Hkv  = static_cast<int64_t>(k.extent(1));
  const int64_t Ktot = static_cast<int64_t>(k.extent(0));
  if (T <= 0 || Hq <= 0 || Hkv <= 0 || D <= 0 || Ktot <= 0) return false;
  if (static_cast<int64_t>(v.extent(0)) != Ktot) return false;
  if (static_cast<int64_t>(v.extent(1)) != Hkv) return false;
  if (static_cast<int64_t>(v.extent(2)) != D) return false;
  if (static_cast<int64_t>(out.extent(0)) != T) return false;
  if (static_cast<int64_t>(out.extent(1)) != Hq) return false;
  if (static_cast<int64_t>(out.extent(2)) != D) return false;
  // GQA requires whole-group mapping, matching the native kernel's
  // hkv = (h * Hkv) / Hq integer projection.
  if (Hq % Hkv != 0) return false;
  // Bottom-right causal matches the native mask iff the KV prefix is
  // exactly the queries' visible keys.
  if (Ktot != static_cast<int64_t>(q_off) + T) return false;
  if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr ||
      out.data() == nullptr)
    return false;

  MiopenSdpaScalar dtype;
  if constexpr (std::is_same_v<Scalar, float>) {
    dtype = MiopenSdpaScalar::Float;
  } else if constexpr (std::is_same_v<Scalar,
                                      Kokkos::Experimental::bhalf_t>) {
    dtype = MiopenSdpaScalar::BFloat16;
  } else {
    return false;
  }

  // No repack (see file note): forward raw Kokkos strides straight to the
  // stub. A future composed path reuses this contract.
  const void* qp = static_cast<const void*>(q.data());
  const void* kp = static_cast<const void*>(k.data());
  const void* vp = static_cast<const void*>(v.data());
  void* op       = static_cast<void*>(out.data());

  const bool sliding = (window > 0);
  return sliding
             ? miopen_sdpa_forward_sliding(
                   handle.miopen_raw(), dtype, qp, kp, vp, op, T, Ktot, Hq,
                   Hkv, D, static_cast<int64_t>(q.stride(0)),
                   static_cast<int64_t>(q.stride(1)),
                   static_cast<int64_t>(q.stride(2)),
                   static_cast<int64_t>(k.stride(0)),
                   static_cast<int64_t>(k.stride(1)),
                   static_cast<int64_t>(k.stride(2)),
                   static_cast<int64_t>(v.stride(0)),
                   static_cast<int64_t>(v.stride(1)),
                   static_cast<int64_t>(v.stride(2)),
                   static_cast<int64_t>(out.stride(0)),
                   static_cast<int64_t>(out.stride(1)),
                   static_cast<int64_t>(out.stride(2)), scale, window)
             : miopen_sdpa_forward(
                   handle.miopen_raw(), dtype, qp, kp, vp, op, T, Ktot, Hq,
                   Hkv, D, static_cast<int64_t>(q.stride(0)),
                   static_cast<int64_t>(q.stride(1)),
                   static_cast<int64_t>(q.stride(2)),
                   static_cast<int64_t>(k.stride(0)),
                   static_cast<int64_t>(k.stride(1)),
                   static_cast<int64_t>(k.stride(2)),
                   static_cast<int64_t>(v.stride(0)),
                   static_cast<int64_t>(v.stride(1)),
                   static_cast<int64_t>(v.stride(2)),
                   static_cast<int64_t>(out.stride(0)),
                   static_cast<int64_t>(out.stride(1)),
                   static_cast<int64_t>(out.stride(2)), scale);
}

// Full causal dispatch: same mask as the native kernel when
// Ktot == q_off + T (checked inside). Currently declines (returns false)
// so the caller runs the native kernel.
template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_miopen_sdpa_gqa(const DNNHandle<Exec>& handle, const QView& q,
                                const KView& k, const VView& v, size_t q_off,
                                const OView& out, float scale) {
  return try_miopen_sdpa_gqa_impl<Scalar>(handle, q, k, v, q_off, out, scale,
                                          -1);
}

// Sliding-window causal dispatch: query t at absolute position p attends
// keys [max(0, p + 1 - sliding_window), p]. sliding_window == 0 declines
// (the caller routes that to try_miopen_sdpa_gqa); otherwise the checks
// match the causal path. Currently declines (returns false) so the caller
// runs the native kernel.
template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_miopen_sdpa_gqa_sliding(const DNNHandle<Exec>& handle,
                                        const QView& q, const KView& k,
                                        const VView& v, size_t q_off,
                                        const OView& out, float scale,
                                        size_t sliding_window) {
  if (sliding_window == 0) return false;
  return try_miopen_sdpa_gqa_impl<Scalar>(handle, q, k, v, q_off, out, scale,
                                          static_cast<int64_t>(sliding_window));
}

// Non-causal (bidirectional) SDPA dispatch over Kokkos views. Mirrors
// try_miopen_sdpa_gqa (dtype mapping, GQA Hq%Hkv check, null-data guards)
// but without the causal Ktot == q_off + T requirement: bidirectional
// attention attends to all K keys, so there is no q_off parameter.
// Currently declines (returns false) so the caller runs the native
// kernel.
template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_miopen_sdpa_bidir(const DNNHandle<Exec>& handle, const QView& q,
                                  const KView& k, const VView& v,
                                  const OView& out, float scale) {
  const int64_t T    = static_cast<int64_t>(q.extent(0));
  const int64_t Hq   = static_cast<int64_t>(q.extent(1));
  const int64_t D    = static_cast<int64_t>(q.extent(2));
  const int64_t Hkv  = static_cast<int64_t>(k.extent(1));
  const int64_t Ktot = static_cast<int64_t>(k.extent(0));
  if (T <= 0 || Hq <= 0 || Hkv <= 0 || D <= 0 || Ktot <= 0) return false;
  if (static_cast<int64_t>(v.extent(0)) != Ktot) return false;
  if (static_cast<int64_t>(v.extent(1)) != Hkv) return false;
  if (static_cast<int64_t>(v.extent(2)) != D) return false;
  if (static_cast<int64_t>(out.extent(0)) != T) return false;
  if (static_cast<int64_t>(out.extent(1)) != Hq) return false;
  if (static_cast<int64_t>(out.extent(2)) != D) return false;
  // GQA requires whole-group mapping, matching the native kernel's
  // hkv = (h * Hkv) / Hq integer projection.
  if (Hq % Hkv != 0) return false;
  if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr ||
      out.data() == nullptr)
    return false;

  MiopenSdpaScalar dtype;
  if constexpr (std::is_same_v<Scalar, float>) {
    dtype = MiopenSdpaScalar::Float;
  } else if constexpr (std::is_same_v<Scalar,
                                      Kokkos::Experimental::bhalf_t>) {
    dtype = MiopenSdpaScalar::BFloat16;
  } else {
    return false;
  }

  return miopen_sdpa_forward_noncausal(
      handle.miopen_raw(), dtype, static_cast<const void*>(q.data()),
      static_cast<const void*>(k.data()), static_cast<const void*>(v.data()),
      static_cast<void*>(out.data()), T, Ktot, Hq, Hkv, D,
      static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
      static_cast<int64_t>(q.stride(2)), static_cast<int64_t>(k.stride(0)),
      static_cast<int64_t>(k.stride(1)), static_cast<int64_t>(k.stride(2)),
      static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
      static_cast<int64_t>(v.stride(2)), static_cast<int64_t>(out.stride(0)),
      static_cast<int64_t>(out.stride(1)), static_cast<int64_t>(out.stride(2)),
      scale);
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
