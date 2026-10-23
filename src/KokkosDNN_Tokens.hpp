// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_TOKENS_HPP
#define KOKKOSDNN_TOKENS_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Impl {

// Batching/layout policy check shared by the batched entry points: every
// extent except the token counts must agree (batch B, channel D); the output
// token count must equal 1 (CLS) + R (registers) + Np (patches). Throws
// std::invalid_argument with the offending extents.
inline void tokens_check_batched(size_t B, size_t D, size_t R, size_t Np,
                                 size_t Bout, size_t Dout, size_t Tout,
                                 const char* what) {
  const std::string w(what);
  if (Bout != B)
    throw std::invalid_argument(w + ": batch mismatch (got output B=" +
                                std::to_string(Bout) + ", expected " +
                                std::to_string(B) + ")");
  if (Dout != D)
    throw std::invalid_argument(w + ": channel mismatch (got output D=" +
                                std::to_string(Dout) + ", expected " +
                                std::to_string(D) + ")");
  if (Tout != 1 + R + Np)
    throw std::invalid_argument(w + ": token count T must equal 1 + R + Np " +
                                "(got T=" + std::to_string(Tout) + ", 1+R+Np=" +
                                std::to_string(1 + R + Np) + ")");
}

// Device copy of one token segment: out(b, t0 + s, d) = seg(b, s, d) over
// (B, S, D). Plain element copy (no accumulation), hence scalar-agnostic.
template <class ExecSpace, class OutView, class SegView>
void tokens_copy_segment(const OutView& out, const SegView& seg, size_t t0,
                         const std::string& label) {
  const size_t B = seg.extent(0), S = seg.extent(1), D = seg.extent(2);
  if (B == 0 || S == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                              {B, S, D}),
      KOKKOS_LAMBDA(size_t b, size_t s, size_t d) {
        out(b, t0 + s, d) = seg(b, s, d);
      });
}

// Device copy of the shared (batch-broadcast) CLS row: out(b, 0, d) = cls(d).
template <class ExecSpace, class OutView, class ClsView>
void tokens_copy_cls_broadcast(const OutView& out, const ClsView& cls,
                               const std::string& label) {
  const size_t B = out.extent(0), D = out.extent(2);
  if (B == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, D}),
      KOKKOS_LAMBDA(size_t b, size_t d) { out(b, 0, d) = cls(d); });
}

// Device copy of the shared (batch-broadcast) register block:
// out(b, 1 + r, d) = regs(r, d) over (B, R, D).
template <class ExecSpace, class OutView, class RegView>
void tokens_copy_regs_broadcast(const OutView& out, const RegView& regs,
                                const std::string& label) {
  const size_t B = out.extent(0), R = regs.extent(0), D = out.extent(2);
  if (B == 0 || R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                              {B, R, D}),
      KOKKOS_LAMBDA(size_t b, size_t r, size_t d) {
        out(b, 1 + r, d) = regs(r, d);
      });
}

}  // namespace Impl

// ViT token assembly: concatenate along the token dim into the full encoder
// sequence
//   out[b, 0, :]     = cls
//   out[b, 1:1+R, :] = registers
//   out[b, 1+R:T, :] = patches,   T = 1 + R + Np.
// Covers the DINOv3 ViT-B/16 case (1 CLS + 4 registers + Np=196 patches at
// 224px/16px -> T=201) as well as general R/Np (R=0 degrades to CLS+patches).
//
// [B,T,D] policy (documented choice): every sequence tensor is rank-3
// [batch, tokens, channels] Kokkos::LayoutRight, so the token dim is the
// middle extent and each (b, t) row is a contiguous D-lane. The token order
// above is fixed: downstream 2D axial RoPE assumes prefix_skip = 1 + R
// untouched rows followed by row-major patches. Scalar-agnostic (plain
// copies); all copies run as device parallel_for kernels, never host loops.
//
// Fully batched form: per-sample prefix (cls [B,1,D], registers [B,R,D]).
template <class ExecSpace, class Scalar>
void tokens_assemble_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& cls,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& registers,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& patches,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& output,
    const std::string& label = "KokkosDNN::tokens_assemble") {
  const size_t B = patches.extent(0), Np = patches.extent(1);
  const size_t D = patches.extent(2);
  if (cls.extent(0) != B || cls.extent(1) != 1 || cls.extent(2) != D)
    throw std::invalid_argument(
        "tokens_assemble_forward: cls must be [B,1,D] (got [" +
        std::to_string(cls.extent(0)) + "," + std::to_string(cls.extent(1)) +
        "," + std::to_string(cls.extent(2)) + "], expected [" +
        std::to_string(B) + ",1," + std::to_string(D) + "])");
  if (registers.extent(0) != B || registers.extent(2) != D)
    throw std::invalid_argument(
        "tokens_assemble_forward: registers must be [B,R,D] (got [" +
        std::to_string(registers.extent(0)) + "," +
        std::to_string(registers.extent(1)) + "," +
        std::to_string(registers.extent(2)) + "], expected B=" +
        std::to_string(B) + ", D=" + std::to_string(D) + ")");
  Impl::tokens_check_batched(B, D, registers.extent(1), Np, output.extent(0),
                             output.extent(2), output.extent(1),
                             "tokens_assemble_forward");
  if (B == 0 || D == 0 || output.extent(1) == 0) return;
  Impl::tokens_copy_segment<ExecSpace>(output, cls, 0, label + "_cls");
  Impl::tokens_copy_segment<ExecSpace>(output, registers, 1,
                                       label + "_registers");
  Impl::tokens_copy_segment<ExecSpace>(output, patches, 1 + registers.extent(1),
                                       label + "_patches");
}

// Shared-prefix form: one CLS vector broadcast across the batch (learned
// parameter, same for every sample), register block [R,D] broadcast the
// same way. Matches HF checkpoints where cls_token / register_tokens are
// stored unbatched.
template <class ExecSpace, class Scalar>
void tokens_assemble_forward(
    const Kokkos::View<Scalar*, ExecSpace>& cls,  // [D]
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& registers,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& patches,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& output,
    const std::string& label = "KokkosDNN::tokens_assemble") {
  const size_t B = patches.extent(0), Np = patches.extent(1);
  const size_t D = patches.extent(2);
  if (cls.extent(0) != D)
    throw std::invalid_argument(
        "tokens_assemble_forward: cls must be [D] (got " +
        std::to_string(cls.extent(0)) + ", expected " + std::to_string(D) +
        ")");
  if (registers.extent(1) != D)
    throw std::invalid_argument(
        "tokens_assemble_forward: registers must be [R,D] (got [" +
        std::to_string(registers.extent(0)) + "," +
        std::to_string(registers.extent(1)) + "], expected D=" +
        std::to_string(D) + ")");
  Impl::tokens_check_batched(B, D, registers.extent(0), Np, output.extent(0),
                             output.extent(2), output.extent(1),
                             "tokens_assemble_forward");
  if (B == 0 || D == 0 || output.extent(1) == 0) return;
  Impl::tokens_copy_cls_broadcast<ExecSpace>(output, cls, label + "_cls");
  Impl::tokens_copy_regs_broadcast<ExecSpace>(output, registers,
                                              label + "_registers");
  Impl::tokens_copy_segment<ExecSpace>(output, patches, 1 + registers.extent(0),
                                       label + "_patches");
}

// Shared-prefix form with the CLS row kept as a [1,D] matrix (HF cls_token
// shape [1,1,D] squeezed once). Otherwise identical to the [D] overload.
template <class ExecSpace, class Scalar>
void tokens_assemble_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cls,  // [1,D]
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& registers,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& patches,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& output,
    const std::string& label = "KokkosDNN::tokens_assemble") {
  const size_t D = patches.extent(2);
  if (cls.extent(0) != 1 || cls.extent(1) != D)
    throw std::invalid_argument(
        "tokens_assemble_forward: cls must be [1,D] (got [" +
        std::to_string(cls.extent(0)) + "," + std::to_string(cls.extent(1)) +
        "], expected [1," + std::to_string(D) + "])");
  // Rank-1 alias over the single row; the kernels below only read it.
  Kokkos::View<Scalar*, ExecSpace> row(cls.data(), D);
  tokens_assemble_forward<ExecSpace, Scalar>(row, registers, patches, output,
                                             label);
}

// Single-sequence convenience form: patches [Np,D] -> output [T,D] with
// T = 1 + R + Np. Same token order and LayoutRight row policy as above.
template <class ExecSpace, class Scalar>
void tokens_assemble_single(
    const Kokkos::View<Scalar*, ExecSpace>& cls,  // [D]
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& registers,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& patches,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& output,
    const std::string& label = "KokkosDNN::tokens_assemble_single") {
  const size_t Np = patches.extent(0), D = patches.extent(1);
  const size_t R = registers.extent(0);
  if (cls.extent(0) != D)
    throw std::invalid_argument(
        "tokens_assemble_single: cls must be [D] (got " +
        std::to_string(cls.extent(0)) + ", expected " + std::to_string(D) +
        ")");
  if (registers.extent(1) != D)
    throw std::invalid_argument(
        "tokens_assemble_single: registers must be [R,D] (got [" +
        std::to_string(registers.extent(0)) + "," +
        std::to_string(registers.extent(1)) + "], expected D=" +
        std::to_string(D) + ")");
  if (output.extent(0) != 1 + R + Np || output.extent(1) != D)
    throw std::invalid_argument(
        "tokens_assemble_single: output must be [1+R+Np,D] (got [" +
        std::to_string(output.extent(0)) + "," +
        std::to_string(output.extent(1)) + "], expected [" +
        std::to_string(1 + R + Np) + "," + std::to_string(D) + "])");
  if (D == 0 || output.extent(0) == 0) return;
  Kokkos::parallel_for(
      label + "_cls", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d) { output(0, d) = cls(d); });
  if (R > 0)
    Kokkos::parallel_for(
        label + "_registers",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
        KOKKOS_LAMBDA(size_t r, size_t d) { output(1 + r, d) = registers(r, d); });
  if (Np > 0)
    Kokkos::parallel_for(
        label + "_patches",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Np, D}),
        KOKKOS_LAMBDA(size_t p, size_t d) {
          output(1 + R + p, d) = patches(p, d);
        });
}

// Single-sequence form with the CLS row as [1,D] (see batched overload).
template <class ExecSpace, class Scalar>
void tokens_assemble_single(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cls,  // [1,D]
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& registers,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& patches,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& output,
    const std::string& label = "KokkosDNN::tokens_assemble_single") {
  const size_t D = patches.extent(1);
  if (cls.extent(0) != 1 || cls.extent(1) != D)
    throw std::invalid_argument(
        "tokens_assemble_single: cls must be [1,D] (got [" +
        std::to_string(cls.extent(0)) + "," + std::to_string(cls.extent(1)) +
        "], expected [1," + std::to_string(D) + "])");
  Kokkos::View<Scalar*, ExecSpace> row(cls.data(), D);
  tokens_assemble_single<ExecSpace, Scalar>(row, registers, patches, output,
                                            label);
}

}  // namespace KokkosDNN

#endif
