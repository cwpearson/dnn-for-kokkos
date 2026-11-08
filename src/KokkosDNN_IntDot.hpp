// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INTDOT_HPP
#define KOKKOSDNN_INTDOT_HPP

// Backend-specialized 4-way signed-int8 dot product with int32 accumulation.
//
// What this abstracts: one integer instruction that computes
//   acc += a0*b0 + a1*b1 + a2*b2 + a3*b3
// over four packed signed-int8 lanes (CUDA: a single `__dp4a`; the MMVQ
// decode GEMV issues ~8 of these per 32 weight elements). The primary
// template is a portable scalar loop with bit-identical results (integer
// arithmetic is exact, so all specializations agree by construction);
// execution spaces with a single-instruction int8 dot specialize it.
//
// Usage: load 4-byte words and dot them. `load` needs a 4-byte-aligned
// address (one load instruction); `load_u` is safe at any alignment for the
// 2-byte-aligned Q6_K/Q8_0 payloads, and `load_u16` is the cheaper form for
// the same payloads when only 2-byte alignment is known (two 16-bit loads
// instead of four byte loads). The natural granularity is one 4-byte word
// per dot, e.g. a serial unrolled loop over words inside one thread:
//
//   int dot = 0;
//   for (int i = 0; i < 4; ++i)
//     dot = KokkosDNN::Impl::IntDot<ExecSpace>::dot4(
//         KokkosDNN::Impl::IntDot<ExecSpace>::load(w + i),
//         KokkosDNN::Impl::IntDot<ExecSpace>::load(x + i), dot);
//
// Correctness notes:
//   * Bytes are interpreted as SIGNED int8 on every path (`__dp4a` semantics).
//     Non-negative payloads (4/5/6-bit quants) are unaffected by signedness;
//     genuinely signed payloads (int8 activations, Q8_0 weights) must be
//     signed, which is what both paths compute.
//   * int32 addition is exact and order-independent, so vector-lane
//     reassociation cannot change the result.
//   * The CUDA path additionally requires sm_61+ (`__CUDA_ARCH__ >= 610`);
//     older architectures and the host pass of the functor transparently get
//     the scalar fallback (still correct, just slower).
//
// Extension point: add `IntDot<YourExecSpace>` specializations the same way
// (HIP `sdot4`, SYCL DP4A, ...) -- callers need no changes.
#include <Kokkos_Core.hpp>

#include <cstdint>

namespace KokkosDNN {
namespace Impl {

namespace IntDotDetail {

// Portable 4-way signed-int8 dot, factored out so the primary template and
// every specialization's fallback branch share one implementation.
KOKKOS_INLINE_FUNCTION int32_t dot4_scalar(int32_t a, int32_t b,
                                           int32_t acc) {
  const auto sx = [](uint32_t u) -> int32_t {
    return (int32_t)(int8_t)(uint8_t)u;
  };
  const uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
  int32_t s = acc;
  s += sx(ua & 0xFFu) * sx(ub & 0xFFu);
  s += sx((ua >> 8) & 0xFFu) * sx((ub >> 8) & 0xFFu);
  s += sx((ua >> 16) & 0xFFu) * sx((ub >> 16) & 0xFFu);
  s += sx((ua >> 24) & 0xFFu) * sx((ub >> 24) & 0xFFu);
  return s;
}

// Aligned 4-byte load: a single 4-byte load instruction. The caller
// guarantees 4-byte alignment (see the contract at each call site: packed
// rows whose block size is a multiple of 16 at a 16-aligned base, and the
// int8 activation row at 16-aligned staging plus 16-multiple offsets).
// Deliberately NOT a memcpy and NOT behind a runtime branch: nvcc lowers a
// 4-byte memcpy to 4 byte loads + permutes even when every dynamic address
// is aligned (measured), and pairing that fallback with a branch poisons the
// aligned path to byte loads too. Two entry points instead of one branch.
KOKKOS_INLINE_FUNCTION int32_t load_aligned32(const void* p) {
  return *reinterpret_cast<const int32_t*>(p);
}

// Unaligned-safe 4-byte load for the 2-byte-aligned payloads (Q6_K's
// 210-byte blocks, Q8_0's 34-byte blocks: odd block indices are 2 mod 4).
// Manual byte assembly -- memcpy-free for the same codegen reason as above:
// 4 byte loads plus merges, correct at every alignment on every backend.
KOKKOS_INLINE_FUNCTION int32_t load_unaligned32(const void* p) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  return (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                   ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24));
}

// 4-byte load for 2-byte-aligned payloads: two natural 16-bit loads instead of
// four byte loads, which is what the unaligned path costs. Used by the Q6_K /
// Q8_0 weight windows (their block sizes are even, so every window is at
// least 2-byte aligned).
KOKKOS_INLINE_FUNCTION int32_t load_u16x2(const void* p) {
  const uint16_t* q = static_cast<const uint16_t*>(p);
  return (int32_t)((uint32_t)q[0] | ((uint32_t)q[1] << 16));
}

}  // namespace IntDotDetail

// Primary template: portable, exact, deliberately dumb.
template <class ExecSpace>
struct IntDot {
  KOKKOS_INLINE_FUNCTION static int32_t dot4(int32_t a, int32_t b,
                                             int32_t acc) {
    return IntDotDetail::dot4_scalar(a, b, acc);
  }
  KOKKOS_INLINE_FUNCTION static int32_t load(const void* p) {
    return IntDotDetail::load_aligned32(p);
  }
  // Unaligned-safe variant for 2-byte-aligned payloads (Q6_K/Q8_0 windows).
  KOKKOS_INLINE_FUNCTION static int32_t load_u(const void* p) {
    return IntDotDetail::load_unaligned32(p);
  }
  // 2-byte-aligned variant: two 16-bit loads (cheaper than load_u).
  KOKKOS_INLINE_FUNCTION static int32_t load_u16(const void* p) {
    return IntDotDetail::load_u16x2(p);
  }
};

#if defined(KOKKOS_ENABLE_CUDA)
// CUDA: one `__dp4a` per 4-element dot on sm_61+ device code; the scalar
// fallback covers the host pass and older architectures.
template <>
struct IntDot<Kokkos::Cuda> {
  KOKKOS_INLINE_FUNCTION static int32_t dot4(int32_t a, int32_t b,
                                             int32_t acc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(a, b, acc);
#else
    return IntDotDetail::dot4_scalar(a, b, acc);
#endif
  }
  KOKKOS_INLINE_FUNCTION static int32_t load(const void* p) {
    return IntDotDetail::load_aligned32(p);
  }
  // Unaligned-safe variant for 2-byte-aligned payloads (Q6_K/Q8_0 windows).
  KOKKOS_INLINE_FUNCTION static int32_t load_u(const void* p) {
    return IntDotDetail::load_unaligned32(p);
  }
  // 2-byte-aligned variant: two 16-bit loads (cheaper than load_u).
  KOKKOS_INLINE_FUNCTION static int32_t load_u16(const void* p) {
    return IntDotDetail::load_u16x2(p);
  }
};
#endif

}  // namespace Impl
}  // namespace KokkosDNN

#endif
