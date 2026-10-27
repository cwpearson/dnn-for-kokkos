// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CONV_HPP
#define KOKKOSDNN_CONV_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"

// MIOpen dispatch (HIP builds with KokkosDNN_ENABLE_MIOPEN only). The
// declarations live in tpl/miopen, mirroring the cuDNN SDPA arrangement;
// the native kernels below are the fallback on every other build.
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
#include "tpl/miopen/KokkosDNN_MIOpen_ConvForward.hpp"
#endif
// cuDNN dispatch (CUDA builds with KokkosDNN_ENABLE_CUDNN only). The
// declarations live in tpl/cudnn, mirroring the cuDNN SDPA arrangement;
// the im2col lowering / native kernels below are the fallback.
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
#include "tpl/cudnn/KokkosDNN_CuDNN_ConvForward.hpp"
#endif

// Spatial convolutions over token-row activations [T, C] with T = H*W
// row-major (t = h*W + w). That layout is exactly N=1 NHWC memory order,
// so TPL dispatches pass device pointers through zero-copy.
//
// Three operators, all stride 1 with symmetric zero padding:
//
//   conv3x3_forward         dense 3x3, pad 1, optional bias
//   depthwise_conv_forward  depthwise KxK (runtime odd K), pad K/2,
//                           optional bias
//   grouped_conv1x1_forward grouped 1x1, no padding, optional bias
//
// Plus one strided operator for VAE encoder downsamplers:
//
//   conv3x3_stride2_forward dense 3x3, stride 2, ZeroPad(0,1,0,1)
//                           (asymmetric: zero past the right/bottom edges
//                           only; H and W must be even), optional bias
//
// Scalar-agnostic (float, bhalf_t); accumulation is FP32, cast back to
// Scalar. The native kernels (im2col gather, depthwise, grouped) are
// backend-portable parallel_for only (no backend-specific code outside
// src/tpl); the dense 3x3's compute goes through the dispatched gemm_bias
// core op.
//
// TPL status: on CUDA builds with KokkosDNN_ENABLE_CUDNN all three ops
// dispatch to cuDNN convolution forward (implicit GEMM / Winograd engines
// via the legacy cudnnConvolution* entry points in the cudnn_cnn
// sublibrary) when handle.use_cudnn is set, falling back to the lowering
// / native kernels below on any dispatch failure. FP32 I/O stays FP32
// (CUDNN_FMA_MATH, same numerics contract as the im2col + sgemm lowering
// and the old native kernel; no TF32); BF16 uses tensor-op math with FP32
// accumulation. The dense 3x3 lowering (im2col + gemm_bias, riding the
// proven cuBLAS ladder) and the skinny-output team kernel remain as the
// fallback path, and stay the primary path on builds without cuDNN.
// (History, investigated 2026-10, see task/p1-vae-cudnn-conv: an earlier
// cuDNN dispatch attempt was built and reverted because the legacy
// cudnnConvolution* entry points live in the cudnn_cnn sublibrary, which
// was absent wherever only the loader plus cudnn_graph shipped,
// failing with CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE; and the
// cudnn-frontend conv_fprop node at the pinned commit had no group-count
// knob (ruling out depthwise/grouped) while its only eligible dense
// engine on sm86 (engine 7, NVRTC-JIT) both silently requires
// C-innermost (KRSC) filters and computes FP32 I/O at TF32 precision
// (~6e-4 relative error vs native). Both blockers are gone with full
// cuDNN sublibraries plus the legacy grouped path.)
//
// HIP diverges from that decision for depthwise/grouped: MIOpen convolution
// (miopenConvolutionForward/Bias + miopenSetConvolutionGroupCount, all in
// the single MIOpen library, no absent sublibrary) fully covers dense,
// depthwise, and grouped forward convolution, so HIP builds with
// KokkosDNN_ENABLE_MIOPEN dispatch depthwise/grouped to
// tpl/miopen/KokkosDNN_MIOpen_ConvForward.* when the handle has MIOpen
// enabled (handle.use_miopen) and fall back to the native kernels below
// otherwise (any dispatch failure also falls back). The dense 3x3 takes
// the MIOpen path when enabled (same knob) and otherwise the im2col +
// gemm lowering, which on HIP rides the hipBLAS ladder. All other builds
// run the lowering (dense) / native kernels (depthwise, grouped)
// unconditionally.
//
// The stride-2 operator has no TPL convolution dispatch on any build, by
// construction rather than by omission: the legacy cudnnConvolution* and
// miopen convolution descriptors each take a single symmetric pad per
// spatial axis, so the asymmetric ZeroPad(0,1,0,1) (right/bottom-only)
// padding of the downsampler is not expressible through them, and a
// symmetric-pad dispatch would silently compute the wrong border values.
// The stride-2 lowering below (im2col-stride2 + gemm_bias, same budget
// blocking) still rides the cuBLAS/hipBLAS ladder for its compute, so TPL
// silicon is used where it matters; only the gather stays native. (A
// future cuDNN-backend-API path could express the asymmetry with pre/post
// paddings; until then native == dispatched by construction.)
//
// Bias convention (matches gemm_bias / depthwise_conv3x3_forward): b is
// [Cout] (or [C] for depthwise), or empty (extent 0, e.g. a
// default-constructed view) for the unbiased path.

namespace KokkosDNN {
namespace Impl {

// Dense 3x3 lowering: im2col + gemm_bias. The old naive kernel here
// parallelized over (T*Cout) with an inner loop over Cin*9 in global
// memory -- 2 FP32 loads per FMA, zero data reuse (nsys: ~5.4 s of a
// 117 ms-budget VAE decode on RTX 3090). Lowering instead materializes
// the [T, Cin*9] im2col matrix (zero-padded gather, one portable kernel)
// and delegates the compute to gemm_bias, which owns the proven
// cuBLAS/hipBLAS ladder (FP32 accumulation, same numerics contract as the
// old kernel) with a portable tiled fallback elsewhere. No backend-gated
// code here: the TPL choice lives entirely inside gemm/gemm_bias.
//
// The im2col scratch is row-blocked to a fixed budget
// (ConvIm2colBudgetBytes) so the 512x512 VAE head layers (T = 262144,
// Cin = 96 -> 906 MB unblocked) never spike device memory: each block is
// gathered then immediately consumed by one gemm_bias call. w [Cout,Cin,3,3] is LayoutRight dense,
// so the unmanaged [Cout, Cin*9] flatten aliases it zero-copy, as do the
// [cur, K] column block (prefix rows of the scratch) and the [cur, Cout]
// output rows (contiguous token rows of y).

// Gather one im2col row-block: col[tm, i*9+kh*3+kw] =
// x_pad((r0+tm)/W + kh - 1, (r0+tm)%W + kw - 1, i), zero outside [H,W).
// col is an unmanaged view over the first `cur` rows of the scratch (or
// the whole scratch when unblocked); x is the full [T, Cin] token input.
//
// Flat 1D RangePolicy with k innermost (e = tm*K9 + k): consecutive GPU
// threads step along k, the LayoutRight-contiguous dimension, so both the
// col writes and (for fixed (kh,kw)) the x reads are coalesced. A Rank<2>
// MDRangePolicy must NOT be used here: on CUDA its default tile steps the
// leftmost index with threadIdx.x, i.e. consecutive threads would step
// along tm (stride K9) -- measured ~8x sector waste on both streams.
template <class ExecSpace, class Scalar>
void conv3x3_im2col_block(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                       Kokkos::MemoryUnmanaged>& col,
    size_t r0, size_t H, size_t W, const std::string& label) {
  const size_t cur = col.extent(0), K9 = col.extent(1);
  if (cur == 0 || K9 == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, cur * K9),
      KOKKOS_LAMBDA(size_t e) {
        const size_t tm = e / K9, k = e % K9;
        const size_t i = k / 9, rem = k % 9;
        const size_t kh = rem / 3, kw = rem % 3;
        const size_t t = r0 + tm;
        const size_t h = t / W, ww = t % W;
        const int ih = static_cast<int>(h) + static_cast<int>(kh) - 1;
        const int iw = static_cast<int>(ww) + static_cast<int>(kw) - 1;
        Scalar v = static_cast<Scalar>(0);
        if (ih >= 0 && static_cast<size_t>(ih) < H && iw >= 0 &&
            static_cast<size_t>(iw) < W)
          v = x(static_cast<size_t>(ih) * W + static_cast<size_t>(iw), i);
        col(tm, k) = v;
      });
}

// Skinny-output direct convolution: one team per output position t, the
// team cooperatively stages the 3x3 patch (K9 = Cin*9 floats, FP32) in
// scratch once, then each output channel reduces its strided dot over the
// patch against its flattened weight row. Used when Cout is small (the
// VAE RGB head: Cout = 3), where the im2col + GEMM lowering is hopeless:
// a skinny-N sgemm cannot fill the machine (measured 0.1 TFLOPS on
// M=262144/N=3/K=864) while this kernel streams ~1 GB of patch data and
// hits L2 on the tiny weight rows. Portable: TeamPolicy + team scratch +
// TeamThreadRange reduce only; the team size is probed at runtime (same
// pattern as Impl::gemv_native) so team-incapable backends still run
// correctly with team size 1.
template <class ExecSpace, class Scalar>
struct SkinnyConvFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using Xu = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  using Wu = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  using Yu = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  using Bu = Kokkos::View<Scalar*, ExecSpace, Kokkos::MemoryUnmanaged>;
  Xu x;
  Wu w2d;  // [Cout, K9] flatten of the dense weight
  Bu b;    // [Cout], possibly empty (extent 0)
  Yu y;
  size_t H, W, K9, shBytes;
  bool biased;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const size_t t = static_cast<size_t>(team.league_rank());
    const int Tk = team.team_size();
    const int rank = team.team_rank();
    float* sh = static_cast<float*>(team.team_scratch(0).get_shmem(shBytes));
    float* patch = sh;      // [K9] staged input patch (FP32)
    float* psum = sh + K9;  // [team_size] per-thread partials
    const size_t h = t / W, ww = t % W;
    // Cooperative gather: consecutive threads take consecutive k, i.e.
    // consecutive channels of (mostly) the same input row -- coalesced.
    for (size_t e = static_cast<size_t>(rank); e < K9;
         e += static_cast<size_t>(Tk)) {
      const size_t i = e / 9, rem = e % 9;
      const size_t kh = rem / 3, kw = rem % 3;
      const int ih = static_cast<int>(h) + static_cast<int>(kh) - 1;
      const int iw = static_cast<int>(ww) + static_cast<int>(kw) - 1;
      float v = 0.0f;
      if (ih >= 0 && static_cast<size_t>(ih) < H && iw >= 0 &&
          static_cast<size_t>(iw) < W)
        v = static_cast<float>(
            x(static_cast<size_t>(ih) * W + static_cast<size_t>(iw), i));
      patch[e] = v;
    }
    team.team_barrier();
    const size_t Cout = w2d.extent(0);
    for (size_t o = 0; o < Cout; ++o) {
      float partial = 0.0f;
      // Strided dot: consecutive threads read consecutive patch/weight
      // elements -- coalesced on both streams (weights hit L2: Cout*K9
      // is a few KB here).
      for (size_t e = static_cast<size_t>(rank); e < K9;
           e += static_cast<size_t>(Tk))
        partial += patch[e] * static_cast<float>(w2d(o, e));
      psum[rank] = partial;
      team.team_barrier();
      float sum = 0.0f;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, Tk),
          KOKKOS_LAMBDA(const int i, float& lsum) { lsum += psum[i]; }, sum);
      if (rank == 0) {
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        y(t, o) = static_cast<Scalar>(acc + sum);
      }
      team.team_barrier();
    }
  }
};

// Skinny-output launcher (Cout <= ConvSkinnyMaxCout and the patch fits in
// team scratch; otherwise the caller uses the lowering). Runtime
// team-size probe first (team-incapable backends clamp to 1; all loops
// stride on the runtime size so any size is correct).
template <class ExecSpace, class Scalar>
void conv_dense3x3_skinny(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t T = x.extent(0), Cout = w.extent(0);
  const size_t K9 = x.extent(1) * 9;
  if (T == 0 || Cout == 0) return;
  using Functor = SkinnyConvFunctor<ExecSpace, Scalar>;
  typename Functor::Xu xu(x.data(), T, x.extent(1));
  typename Functor::Wu wu(w.data(), Cout, K9);
  typename Functor::Bu bu(const_cast<Scalar*>(b.data()), b.extent(0));
  typename Functor::Yu yu(y.data(), T, Cout);
  const bool biased = (b.extent(0) != 0);
  int team_size = 128;
  const size_t shBytes = (K9 + static_cast<size_t>(team_size)) * sizeof(float);
  {
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(shBytes));
    Functor f{xu, wu, bu, yu, H, W, K9, shBytes, biased};
    int mx = probe.team_size_max(f, Kokkos::ParallelForTag{});
    if (mx < team_size) team_size = mx;
    if (team_size < 1) team_size = 1;
  }
  const size_t shBytesRt =
      (K9 + static_cast<size_t>(team_size)) * sizeof(float);
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(T), team_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(shBytesRt));
  Functor f{xu, wu, bu, yu, H, W, K9, shBytesRt, biased};
  Kokkos::parallel_for(label, policy, f);
}

// Tunables (documented, not frozen: tests check functionality, not values).
// Row-blocked im2col scratch budget: peak transient device memory per
// conv3x3 call. 1 GB unblocks every VAE-decode GEMM except the two
// 512x512 / Cin=192 layers (2 row-blocks each), keeping each sgemm at
// large M (chunking measurably costs sgemm efficiency: full-M 16.6 vs
// half-M 9.7 TFLOPS on the 512x512/96-wide layer) while staying a small
// fraction of device memory.
inline constexpr size_t ConvIm2colBudgetBytes = 1024u * 1024u * 1024u;
// Max output channels for the skinny team kernel: at or below this the
// im2col + GEMM lowering is sgemm-efficiency-bound (tiny N) and the
// single-pass team kernel wins; above it GEMM's compute density wins.
inline constexpr size_t ConvSkinnyMaxCout = 32;
// Max flattened patch (Cin*9) for the skinny team kernel, in elements:
// the FP32 patch must fit comfortably in team scratch next to the
// per-thread partials (48 KB budget). Larger patches use the lowering.
inline constexpr size_t ConvSkinnyMaxK9 = 8192;

// Dense 3x3, stride 1, pad 1 (zero padding). Rank-2 token path: x [T,Cin]
// with T = H*W row-major, w [Cout,Cin,3,3], b [Cout] (or empty), y [T,Cout].
template <class ExecSpace, class Scalar>
void conv_dense3x3_2d(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  if (T == 0 || Cin == 0 || Cout == 0) return;
  const size_t K9 = Cin * 9;
  // Skinny-output fast path (tiny N kills sgemm efficiency).
  if (Cout <= ConvSkinnyMaxCout && K9 <= ConvSkinnyMaxK9) {
    conv_dense3x3_skinny<ExecSpace, Scalar>(x, w, b, y, H, W, label);
    return;
  }
  // Row-block the scratch so peak im2col memory stays at the budget.
  size_t Tm = T;
  const size_t per_row = K9 * sizeof(Scalar);
  if (per_row != 0) {
    size_t budget_rows = ConvIm2colBudgetBytes / per_row;
    if (budget_rows < 1) budget_rows = 1;
    if (Tm > budget_rows) Tm = budget_rows;
  }
  using Mat2u = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                             Kokkos::MemoryUnmanaged>;
  // Zero-copy [Cout, Cin*9] flatten of the dense LayoutRight weight.
  Mat2u w2d(w.data(), Cout, K9);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> col("conv3x3_im2col",
                                                             Tm, K9);
  for (size_t r0 = 0; r0 < T; r0 += Tm) {
    const size_t cur = (r0 + Tm <= T) ? Tm : T - r0;
    Mat2u col_cur(col.data(), cur, K9);
    Mat2u y_cur(y.data() + r0 * Cout, cur, Cout);
    conv3x3_im2col_block<ExecSpace, Scalar>(x, col_cur, r0, H, W,
                                            label + "_im2col");
    ::KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, col_cur, w2d, b, y_cur,
                                              label + "_gemm");
  }
}

// Gather one stride-2 im2col row-block: col[tm, i*9+kh*3+kw] =
// x_pad(2*h2+kh, 2*w2+kw, i) with ZeroPad(0,1,0,1) semantics (zero past
// the right/bottom edges), where (h2, w2) is the output position of row
// r0+tm. Same flat-1D, k-innermost layout as conv3x3_im2col_block (see
// the coalescing note there); H2/W2 are the output dims (H/2, W/2).
template <class ExecSpace, class Scalar>
void conv3x3_stride2_im2col_block(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                       Kokkos::MemoryUnmanaged>& col,
    size_t r0, size_t H, size_t W, size_t H2, size_t W2,
    const std::string& label) {
  const size_t cur = col.extent(0), K9 = col.extent(1);
  if (cur == 0 || K9 == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, cur * K9),
      KOKKOS_LAMBDA(size_t e) {
        const size_t tm = e / K9, k = e % K9;
        const size_t i = k / 9, rem = k % 9;
        const size_t kh = rem / 3, kw = rem % 3;
        const size_t t2 = r0 + tm;
        const size_t h2 = t2 / W2, w2 = t2 % W2;
        const size_t ih = 2 * h2 + kh, iw = 2 * w2 + kw;
        Scalar v = static_cast<Scalar>(0);
        if (ih < H && iw < W) v = x(ih * W + iw, i);
        col(tm, k) = v;
      });
  (void)H2;
}

// Skinny-output direct stride-2 convolution: one output element per
// thread, FP32 accumulation over the Cin*9 patch gathered on the fly with
// the same ZeroPad(0,1,0,1) semantics. Used when Cout is small (same
// cutoff as the stride-1 skinny team kernel), where the im2col + GEMM
// lowering is sgemm-efficiency-bound (tiny N); the single-pass kernel
// avoids materializing im2col entirely. Portable parallel_for only.
template <class ExecSpace, class Scalar>
void conv_dense3x3_stride2_skinny(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, size_t H2, size_t W2, const std::string& label) {
  const size_t T2 = y.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (T2 == 0 || Cout == 0) return;
  // Output-channel innermost: consecutive threads write consecutive
  // channels of one token row (LayoutRight-contiguous).
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T2 * Cout),
      KOKKOS_LAMBDA(size_t e) {
        const size_t t2 = e / Cout, o = e % Cout;
        const size_t h2 = t2 / W2, w2 = t2 % W2;
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        for (size_t i = 0; i < Cin; ++i)
          for (size_t kh = 0; kh < 3; ++kh) {
            const size_t ih = 2 * h2 + kh;
            if (ih >= H) continue;
            for (size_t kw = 0; kw < 3; ++kw) {
              const size_t iw = 2 * w2 + kw;
              if (iw >= W) continue;
              acc += static_cast<float>(x(ih * W + iw, i)) *
                     static_cast<float>(w(o, i, kh, kw));
            }
          }
        y(t2, o) = static_cast<Scalar>(acc);
      });
  (void)H2;
}

// Dense 3x3, stride 2, ZeroPad(0,1,0,1) (asymmetric: zero past the
// right/bottom edges only; H and W must be even). Rank-2 token path:
// x [T,Cin] with T = H*W row-major, w [Cout,Cin,3,3], b [Cout] (or
// empty), y [T2,Cout] with T2 = (H/2)*(W/2). Same lowering discipline as
// conv_dense3x3_2d (skinny direct kernel, else budget-blocked im2col +
// gemm_bias reusing ConvIm2colBudgetBytes), so the compute still rides
// the dispatched gemm ladder on every backend.
template <class ExecSpace, class Scalar>
void conv_dense3x3_stride2_2d(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  if (T == 0 || Cin == 0 || Cout == 0) return;
  const size_t H2 = H / 2, W2 = W / 2;
  const size_t K9 = Cin * 9;
  // Skinny-output fast path (tiny N kills sgemm efficiency).
  if (Cout <= ConvSkinnyMaxCout && K9 <= ConvSkinnyMaxK9) {
    conv_dense3x3_stride2_skinny<ExecSpace, Scalar>(x, w, b, y, H, W, H2, W2,
                                                   label);
    return;
  }
  // Row-block the scratch so peak im2col memory stays at the budget.
  const size_t T2 = H2 * W2;
  size_t Tm = T2;
  const size_t per_row = K9 * sizeof(Scalar);
  if (per_row != 0) {
    size_t budget_rows = ConvIm2colBudgetBytes / per_row;
    if (budget_rows < 1) budget_rows = 1;
    if (Tm > budget_rows) Tm = budget_rows;
  }
  using Mat2u = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                             Kokkos::MemoryUnmanaged>;
  // Zero-copy [Cout, Cin*9] flatten of the dense LayoutRight weight.
  Mat2u w2d(w.data(), Cout, K9);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> col(
      "conv3x3_stride2_im2col", Tm, K9);
  for (size_t r0 = 0; r0 < T2; r0 += Tm) {
    const size_t cur = (r0 + Tm <= T2) ? Tm : T2 - r0;
    Mat2u col_cur(col.data(), cur, K9);
    Mat2u y_cur(y.data() + r0 * Cout, cur, Cout);
    conv3x3_stride2_im2col_block<ExecSpace, Scalar>(x, col_cur, r0, H, W, H2,
                                                   W2, label + "_im2col");
    ::KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, col_cur, w2d, b, y_cur,
                                              label + "_gemm");
  }
}

// Depthwise KxK, stride 1, pad K/2 (zero padding), groups = C. Rank-2
// token path: x/y [T,C] with T = H*W row-major, w [C,K,K], b [C] (or
// empty). K must be odd (checked by the caller) so the padding is
// symmetric and the native kernel agrees with the cuDNN dispatch.
template <class ExecSpace, class Scalar>
void conv_depthwise_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t K,
    size_t H, size_t W, const std::string& label) {
  const size_t T = x.extent(0), C = x.extent(1);
  const int P = static_cast<int>(K / 2);
  const bool biased = (b.extent(0) != 0);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (size_t kh = 0; kh < K; ++kh) {
          const int ih = static_cast<int>(h) + static_cast<int>(kh) - P;
          if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
          for (size_t kw = 0; kw < K; ++kw) {
            const int iw = static_cast<int>(ww) + static_cast<int>(kw) - P;
            if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
            acc += static_cast<float>(
                       x(static_cast<size_t>(ih) * W + static_cast<size_t>(iw),
                         c)) *
                   static_cast<float>(w(c, kh, kw));
          }
        }
        y(t, c) = static_cast<Scalar>(acc);
      });
}

// Grouped 1x1 convolution (no padding). x [T,Cin], w [Cout,Cg] (Cg =
// per-group input channels), b [Cout] (or empty), y [T,Cout]. Group g
// owns outputs [g*Co..(g+1)*Co) and inputs [g*Cg..(g+1)*Cg) with
// Co = Cout/groups.
template <class ExecSpace, class Scalar>
void conv_grouped1x1_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    size_t groups, const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const size_t Cg = w.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (T == 0 || Cin == 0 || Cout == 0) return;
  const size_t Co = Cout / groups;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Cout}),
      KOKKOS_LAMBDA(size_t t, size_t o) {
        const size_t g = o / Co;
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        for (size_t k = 0; k < Cg; ++k)
          acc += static_cast<float>(x(t, g * Cg + k)) *
                 static_cast<float>(w(o, k));
        y(t, o) = static_cast<Scalar>(acc);
      });
}

}  // namespace Impl

// Dense 3x3 convolution forward, stride 1, pad 1 (zero padding):
//   y(t,o) = b(o) + sum_{i,kh,kw} w(o,i,kh,kw) * x_pad(h+kh-1, w+kw-1, i)
// with t = h*W + w. x [T,Cin] with T = H*W, w [Cout,Cin,3,3], b [Cout]
// or empty for no bias, y [T,Cout]. Scalar-agnostic; FP32 accumulation.
//
// NOTE: the handle selects the cuDNN path on CUDA builds with
// KokkosDNN_ENABLE_CUDNN and the MIOpen path on HIP builds with
// KokkosDNN_ENABLE_MIOPEN, and is threaded into the im2col + gemm_bias
// lowering otherwise (cuBLAS ladder on CUDA, hipBLAS on HIP, portable
// tiled GEMM elsewhere). The handle is still taken on every build for API
// uniformity with the other operators.
template <class ExecSpace, class Scalar>
void conv3x3_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::conv3x3_forward") {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (w.extent(1) != Cin || w.extent(2) != 3 || w.extent(3) != 3 ||
      y.extent(0) != T || y.extent(1) != Cout ||
      (biased && b.extent(0) != Cout))
    throw std::invalid_argument("conv3x3_forward: dimension mismatch");
  if (H == 0 || W == 0 || T != H * W)
    throw std::invalid_argument("conv3x3_forward: H and W must be nonzero "
                                "with H*W == rows");
  if (T == 0 || Cin == 0 || Cout == 0) return;
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
  constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip) {
    if (handle.use_miopen &&
        Impl::try_miopen_conv3x3_forward<Scalar>(handle, x, w, b, y, H, W))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda) {
    if (handle.use_cudnn &&
        Impl::try_cudnn_conv3x3_forward<Scalar>(handle, x, w, b, y, H, W))
      return;
  }
#endif
  Impl::conv_dense3x3_2d<ExecSpace, Scalar>(handle, x, w, b, y, H, W, label);
}

// Dense 3x3 convolution forward, stride 2, ZeroPad(0,1,0,1) (asymmetric:
// zero past the right/bottom edges only; H and W must be even):
//   y(t2,o) = b(o) + sum_{i,kh,kw} w(o,i,kh,kw) * x_pad(2*h2+kh, 2*w2+kw, i)
// with t2 = h2*W2 + w2, H2 = H/2, W2 = W/2. x [T,Cin] with T = H*W,
// w [Cout,Cin,3,3], b [Cout] or empty for no bias, y [T2,Cout] with
// T2 = H2*W2. Scalar-agnostic; FP32 accumulation.
//
// NOTE: there is deliberately no TPL convolution dispatch here (see the
// TPL-status note at the top of this file): the legacy cuDNN/MIOpen
// descriptors only express symmetric padding, so the asymmetric
// right/bottom-only padding is not expressible through them. The compute
// still rides the dispatched gemm ladder via the im2col + gemm_bias
// lowering. The handle is threaded into that lowering (cuBLAS ladder on
// CUDA, hipBLAS on HIP, portable tiled GEMM elsewhere) and is still
// taken on every build for API uniformity with the other operators.
template <class ExecSpace, class Scalar>
void conv3x3_stride2_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::conv3x3_stride2_forward") {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (H == 0 || W == 0 || (H % 2) != 0 || (W % 2) != 0)
    throw std::invalid_argument("conv3x3_stride2_forward: H and W must be "
                                "even and nonzero");
  const size_t H2 = H / 2, W2 = W / 2;
  if (w.extent(1) != Cin || w.extent(2) != 3 || w.extent(3) != 3 ||
      y.extent(0) != H2 * W2 || y.extent(1) != Cout ||
      (biased && b.extent(0) != Cout))
    throw std::invalid_argument("conv3x3_stride2_forward: dimension mismatch");
  if (T != H * W)
    throw std::invalid_argument("conv3x3_stride2_forward: H and W must be "
                                "nonzero with H*W == input rows");
  if (T == 0 || Cin == 0 || Cout == 0) return;
  Impl::conv_dense3x3_stride2_2d<ExecSpace, Scalar>(handle, x, w, b, y, H, W,
                                                   label);
}

// Depthwise convolution forward, stride 1, symmetric zero padding, one
// KxK kernel per channel (groups = C):
//   y(t,c) = b(c) + sum_{kh,kw} w(c,kh,kw) * x_pad(h+kh-P, w+kw-P, c)
// with t = h*W + w and P = K/2. x/y [T,C] with T = H*W, w [C,K,K],
// b [C] or empty for no bias. K must be odd (even K has no symmetric
// padding, so the call is rejected). Scalar-agnostic; FP32 accumulation.
//
// NOTE: the handle selects the cuDNN path on CUDA builds with
// KokkosDNN_ENABLE_CUDNN and the MIOpen path on HIP builds with
// KokkosDNN_ENABLE_MIOPEN; otherwise the native kernel runs.
template <class ExecSpace, class Scalar>
void depthwise_conv_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t K,
    size_t H, size_t W,
    const std::string& label = "KokkosDNN::depthwise_conv_forward") {
  const size_t T = x.extent(0), C = x.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (K == 0 || K % 2 == 0)
    throw std::invalid_argument(
        "depthwise_conv_forward: K must be a positive odd kernel size");
  if (w.extent(0) != C || w.extent(1) != K || w.extent(2) != K ||
      y.extent(0) != T || y.extent(1) != C || (biased && b.extent(0) != C))
    throw std::invalid_argument("depthwise_conv_forward: dimension mismatch");
  if (H == 0 || W == 0 || T != H * W)
    throw std::invalid_argument("depthwise_conv_forward: H and W must be "
                                "nonzero with H*W == rows");
  if (T == 0 || C == 0) return;
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
  constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip) {
    if (handle.use_miopen &&
        Impl::try_miopen_depthwise_forward<Scalar>(handle, x, w, b, y, K, H,
                                                   W))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda_dw = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda_dw) {
    if (handle.use_cudnn &&
        Impl::try_cudnn_depthwise_forward<Scalar>(handle, x, w, b, y, K, H,
                                                  W))
      return;
  }
#endif
  Impl::conv_depthwise_2d<ExecSpace, Scalar>(x, w, b, y, K, H, W, label);
}

// Grouped 1x1 convolution forward (pointwise, no padding):
//   y(t,o) = b(o) + sum_k w(o,k) * x(t, g*Cg+k),  g = o/(Cout/groups)
// x [T,Cin], w [Cout,Cg] with Cg = Cin/groups, b [Cout] or empty for no
// bias, y [T,Cout]. Cout and Cin must be divisible by groups.
// Scalar-agnostic; FP32 accumulation.
//
// NOTE: the handle selects the cuDNN path on CUDA builds with
// KokkosDNN_ENABLE_CUDNN and the MIOpen path on HIP builds with
// KokkosDNN_ENABLE_MIOPEN; otherwise the native kernel runs.
template <class ExecSpace, class Scalar>
void grouped_conv1x1_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    size_t groups,
    const std::string& label = "KokkosDNN::grouped_conv1x1_forward") {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const size_t Cg = w.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (groups == 0)
    throw std::invalid_argument(
        "grouped_conv1x1_forward: groups must be nonzero");
  if (Cout % groups != 0 || Cin != groups * Cg || y.extent(0) != T ||
      y.extent(1) != Cout || (biased && b.extent(0) != Cout))
    throw std::invalid_argument(
        "grouped_conv1x1_forward: dimension mismatch");
  if (T == 0 || Cin == 0 || Cout == 0) return;
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
  constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip) {
    if (handle.use_miopen &&
        Impl::try_miopen_grouped1x1_forward<Scalar>(handle, x, w, b, y,
                                                    groups))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda_g = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda_g) {
    if (handle.use_cudnn &&
        Impl::try_cudnn_grouped1x1_forward<Scalar>(handle, x, w, b, y,
                                                   groups))
      return;
  }
#endif
  Impl::conv_grouped1x1_2d<ExecSpace, Scalar>(x, w, b, y, groups, label);
}

}  // namespace KokkosDNN

#endif
