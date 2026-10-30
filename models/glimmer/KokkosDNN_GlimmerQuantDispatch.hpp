// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_QUANT_DISPATCH_HPP
#define KOKKOSDNN_GLIMMER_QUANT_DISPATCH_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Glimmer {
namespace Impl {

// Weight-kind trait: true for packed GGUF weights, false for dense views.
// Model::ensure_deq_staging() keys off this (via the block weight type) so
// dense builds rent no staging while packed builds rent exact-size buffers.
// The linear/embed/head helpers below overload on the weight type instead,
// so Model call sites are weight-kind agnostic: swapping Weights to packed
// fields of the SAME names needs no call-site change.
//
// Calling convention (mirrors gemm/gemv/embedding_gather): like every other
// core op these take concrete (managed) view types, so callers passing
// subviews (which carry an explicit Unmanaged trait) must give explicit
// template arguments -- Impl::linear_chunk<ExecSpace, Scalar>(...) -- and
// rely on the usual implicit view conversion, exactly as the former
// gemm<ExecSpace, Scalar>(...) call sites did.
template <class T>
struct is_packed_weight : std::false_type {};

template <class ExecSpace>
struct is_packed_weight<Gguf::QuantWeight<ExecSpace>> : std::true_type {};

template <class T>
inline constexpr bool is_packed_weight_v = is_packed_weight<T>::value;

// ---- Dense linears (current Weights; staging kept for overload symmetry) --

template <class ExecSpace, class Scalar>
void linear_prefill(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Glimmer::linear_prefill") {
  // Dense weights need no staging; the parameter only keeps this overload
  // shape-compatible with the packed one below.
  (void)staging;
  gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), X, W,
                          static_cast<Scalar>(0), Y, label);
}

template <class ExecSpace, class Scalar>
void linear_decode(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Glimmer::linear_decode") {
  if (X.extent(0) != 1 || Y.extent(0) != 1)
    throw std::invalid_argument("linear_decode: expected single-row X and Y");
  (void)staging;
  // M == 1 reuses the native team-per-row GEMV kernel (or cuBLAS M=1) via
  // the shared gemm dispatch; numerics identical to the former call sites.
  gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), X, W,
                          static_cast<Scalar>(0), Y, label);
}

// ---- Packed linears (QuantWeight; decode vs prefill split) ----------------
// Decode (M == 1): on-the-fly quant_gemv, one team per output row, FP32
// accumulation. Prefill (M > 1): dequantize-once into exact staging + dense
// gemm (cuBLAS/TPL dispatch, FP32 accumulation). The on-the-fly quant_gemm
// re-dequantizes each packed weight row once per output row (M times) and
// reloads each activation row once per output column (N times); on a
// 902-token prefill that is 99% of GPU kernel time (nsys, LFM2.5 MoE
// precedent), so prefill pays one dequant + dense GEMMs instead.
// `staging` must be exactly [W.rows, W.cols]: the cuBLAS path requires row
// pitch == K, so carved subviews of oversized buffers would silently fall
// back to the native kernel. The Model rents one exact-size buffer per
// distinct projection shape for this reason.

template <class ExecSpace, class Scalar>
void linear_prefill(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Gguf::QuantWeight<ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Glimmer::linear_prefill") {
  if (staging.extent(0) != W.rows || staging.extent(1) != W.cols)
    throw std::invalid_argument("linear_prefill: staging shape mismatch");
  Gguf::dequant_to_dense<ExecSpace, Scalar>(W, staging);
  gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), X, staging,
                          static_cast<Scalar>(0), Y, label);
}

template <class ExecSpace, class Scalar>
void linear_decode(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Gguf::QuantWeight<ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Glimmer::linear_decode") {
  if (X.extent(0) != 1 || Y.extent(0) != 1)
    throw std::invalid_argument("linear_decode: expected single-row X and Y");
  if (X.extent(1) != W.cols || Y.extent(1) != W.rows)
    throw std::invalid_argument("linear_decode: dimension mismatch");
  (void)staging;  // decode never materializes dense weights
  if (W.rows == 0 || W.cols == 0) return;
  // Unmanaged rank-1 aliases over the single rows: no allocation, and the
  // row pitches are trivially packed (quant_gemv's contract).
  Kokkos::View<Scalar*, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      x1(X.data(), W.cols);
  Kokkos::View<Scalar*, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      y1(Y.data(), W.rows);
  Gguf::quant_gemv<ExecSpace, Scalar>(handle, x1, W, y1, label);
}

// Chunk entry point: decode for single-row chunks, prefill otherwise.
// Weight-kind overloads above select the implementation; the Model calls
// this unconditionally so its loop is identical for dense and packed builds.
template <class ExecSpace, class Scalar, class Weight>
void linear_chunk(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Weight& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Glimmer::linear_chunk") {
  if (X.extent(0) == 1) {
    linear_decode(handle, X, W, staging, Y, label);
  } else {
    linear_prefill(handle, X, W, staging, Y, label);
  }
}

// ---- lm_head (always a single row: rank-1 path both ways) ------------------

template <class ExecSpace, class Scalar>
void lm_head_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar*, ExecSpace>& last,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& logits,
    const std::string& label = "lm_head") {
  gemv<ExecSpace, Scalar>(handle, last, W, logits, label);
}

template <class ExecSpace, class Scalar>
void lm_head_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar*, ExecSpace>& last,
    const Gguf::QuantWeight<ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& logits,
    const std::string& label = "lm_head") {
  Gguf::quant_gemv<ExecSpace, Scalar>(handle, last, W, logits, label);
}

// ---- Embedding gather (dense table vs packed-row dequant) -----------------

template <class ExecSpace, class Scalar>
void embed_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table,
    const Kokkos::View<int*, ExecSpace>& tokens,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "Glimmer::embed") {
  embedding_gather<ExecSpace, Scalar>(table, tokens, out, label);
}

// Packed-row gather: out[t,d] = dequant(table[token[t]], d) via the shared
// device element decoder (same values as dequant_to_dense, FP32 -> Scalar).
// One row is read per token (no M-times redundancy as in quant_gemm), so
// on-the-fly decode is the right trade here; no staging needed. Token ids
// must be in range, exactly as with the dense embedding_gather above.
// Fallback note: if the Weights stream keeps `embed` dense while packing
// the linears, the dense overload above keeps working unchanged.
template <class ExecSpace, class Scalar>
void embed_forward(
    const Gguf::QuantWeight<ExecSpace>& table,
    const Kokkos::View<int*, ExecSpace>& tokens,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "Glimmer::embed_quant") {
  const size_t T = tokens.extent(0);
  const size_t D = table.cols;
  if (out.extent(0) != T || out.extent(1) != D)
    throw std::invalid_argument("embed_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  const uint8_t* data  = table.data.data();
  const size_t rb      = table.row_bytes();
  const Gguf::GgmlType dtype = table.dtype;
  auto toks            = tokens;
  auto dst             = out;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        const size_t row = static_cast<size_t>(toks(t));
        dst(t, d)        = static_cast<Scalar>(
            Gguf::Impl::quant_elem(dtype, data + row * rb, d));
      });
}

}  // namespace Impl
}  // namespace Glimmer
}  // namespace KokkosDNN

#endif
