// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MACROS_HPP
#define KOKKOSDNN_MACROS_HPP

// Portable loop-unroll pragma.
//
// Bare `#pragma unroll` is an nvcc-ism: GCC rejects it with
// -Wunknown-pragmas (an error under KokkosDNN_ENABLE_COMPILER_WARNINGS),
// and MSVC warns (C4068) as well. These macros expand to the spelling each
// compiler accepts:
//
//   KOKKOSDNN_PRAGMA_UNROLL()    full unroll (like bare `#pragma unroll`)
//   KOKKOSDNN_PRAGMA_UNROLL_N(n) unroll `n` times
//
// On compilers without a matching spelling both expand to nothing, so the
// loop still compiles (just without the hint).
#define KOKKOSDNN_IMPL_DO_PRAGMA(x) _Pragma(#x)

#if defined(__CUDA_ARCH__)
// nvcc device pass: `#pragma unroll [n]` is natively supported.
#define KOKKOSDNN_PRAGMA_UNROLL() KOKKOSDNN_IMPL_DO_PRAGMA(unroll)
#define KOKKOSDNN_PRAGMA_UNROLL_N(n) KOKKOSDNN_IMPL_DO_PRAGMA(unroll n)
#elif defined(__CUDACC__)
// nvcc host pass (__CUDACC__ without __CUDA_ARCH__): the host compiler
// (GCC) rejects bare `#pragma unroll` while cudafe warns (#1675-D) on
// `#pragma GCC ...`, so emit nothing. No loss: the host copy of a
// __host__ __device__ loop body is never the hot path.
#define KOKKOSDNN_PRAGMA_UNROLL()
#define KOKKOSDNN_PRAGMA_UNROLL_N(n)
#elif defined(__clang__)
// Clang/AppleClang/hipcc/Intel-LLVM: `#pragma unroll [n]` is supported.
#define KOKKOSDNN_PRAGMA_UNROLL() KOKKOSDNN_IMPL_DO_PRAGMA(unroll)
#define KOKKOSDNN_PRAGMA_UNROLL_N(n) KOKKOSDNN_IMPL_DO_PRAGMA(unroll n)
#elif defined(__HIP_DEVICE_COMPILE__)
// HIP device code without __clang__ (defensive; hipcc is Clang-based).
#define KOKKOSDNN_PRAGMA_UNROLL() KOKKOSDNN_IMPL_DO_PRAGMA(unroll)
#define KOKKOSDNN_PRAGMA_UNROLL_N(n) KOKKOSDNN_IMPL_DO_PRAGMA(unroll n)
#elif defined(__GNUC__)
// GCC needs `#pragma GCC unroll n` with an explicit count. The no-count
// form maps to the maximum count, i.e. full unrolling of any realistic
// loop, matching the bare-`#pragma unroll` intent.
#define KOKKOSDNN_PRAGMA_UNROLL() \
  KOKKOSDNN_IMPL_DO_PRAGMA(GCC unroll 65534)
#define KOKKOSDNN_PRAGMA_UNROLL_N(n) \
  KOKKOSDNN_IMPL_DO_PRAGMA(GCC unroll n)
#else
#define KOKKOSDNN_PRAGMA_UNROLL()
#define KOKKOSDNN_PRAGMA_UNROLL_N(n)
#endif

#endif  // KOKKOSDNN_MACROS_HPP
