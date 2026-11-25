// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_QUANT_DISPATCH_HPP
#define KOKKOSDNN_INKLING_QUANT_DISPATCH_HPP

// Weight-kind-agnostic linear dispatch for the Inkling 2 Mini text decoder
// (models/inkling).
//
// Every 2-D projection in the model (attention q/k/v/gate/out, rel-bias
// linear, sink, router, MoE experts) goes through these helpers, so the
// Model loop is identical for dense and packed builds:
//
//   * dense  weights: plain device Views (row-major [N, K]: Y[M,N]=X[M,K]@W)
//   * packed weights: Gguf::QuantWeight (Q8_0 linears, IQ4_NL/Q8_0 experts)
//
// Decode chunks (M == 1) run on-the-fly quant_gemv on packed weights;
// prefill chunks (M > 1) dequantize once into exact-size staging and run
// the dense gemm path (cuBLAS/TPL dispatch, FP32 accumulation). Dense
// weights take the plain gemm path in both cases (staging ignored).
// Staging must be exactly [W.rows, W.cols]: the cuBLAS path requires
// weight row pitch == K, so carved subviews of oversized buffers would
// silently fall back to the native kernel. The Model rents one exact-size
// staging buffer per distinct projection shape for this reason.
//
// Launch accounting (decode, packed): each linear_chunk decode is two
// launches (quantize act row + int8-dot GEMV) via Gguf::quant_gemv. A layer
// visit shares inputs at the same K (hn feeds wq + wrel + wsink +
// wgate_attn; kh feeds wk + kdyn; vh feeds wv + vdyn), so the Model loop
// quantizes each shared input once (quantize_shared_act below) and reuses
// the staging across its linears (linear_decode_reuse / linear_chunk_reuse
// below, via GgufMvq::gemv_int8_reuse_act stage-2-only). Saves one quantize
// launch per reused linear with identical values (same kernels, only launch
// boundaries move). Grouping is single-buffer safe: each input's linears
// run back-to-back while its quantization occupies act_scratch. Declined
// inputs (unsupported dtype/stride/align, undersized scratch) fall back to
// the regular per-GEMV path, preserving the portable decline rules.
//
// Calling convention (mirrors gemm/gemv/embedding_gather): these take
// concrete (managed) view types, so callers passing subviews (which carry
// an explicit Unmanaged trait) must give explicit template arguments --
// Impl::linear_chunk<ExecSpace, Scalar>(...) -- and rely on the usual
// implicit view conversion, exactly as the former gemm<ExecSpace, Scalar>
// call sites did.

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
namespace Inkling {
namespace Impl {

// Weight-kind trait: true for packed GGUF weights, false for dense views.
// The Model keys staging allocation off this (via the block weight type)
// so dense builds rent no staging while packed builds rent exact-size
// buffers. The linear/embed/head helpers below overload on the weight type
// instead, so Model call sites are weight-kind agnostic.
template <class T>
struct is_packed_weight : std::false_type {};

template <class ExecSpace>
struct is_packed_weight<Gguf::QuantWeight<ExecSpace>> : std::true_type {};

template <class T>
inline constexpr bool is_packed_weight_v = is_packed_weight<T>::value;

// ---- Dense linears (staging kept for overload symmetry) ------------------

template <class ExecSpace, class Scalar>
void linear_prefill(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Inkling::linear_prefill") {
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
    const std::string& label = "Inkling::linear_decode",
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch =
        Kokkos::View<uint8_t*, ExecSpace>()) {
  if (X.extent(0) != 1 || Y.extent(0) != 1)
    throw std::invalid_argument("linear_decode: expected single-row X and Y");
  (void)staging;
  (void)act_scratch;  // dense weights need no int8 staging
  // M == 1 reuses the native team-per-row GEMV kernel (or cuBLAS M=1) via
  // the shared gemm dispatch; numerics identical to the former call sites.
  gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), X, W,
                          static_cast<Scalar>(0), Y, label);
}

// ---- Packed linears (QuantWeight; decode vs prefill split) ----------------
// Decode (M == 1): on-the-fly quant_gemv, one team per output row, FP32
// accumulation. Prefill (M > 1): dequantize-once into exact staging + dense
// gemm (cuBLAS/TPL dispatch, FP32 accumulation).
// `staging` must be exactly [W.rows, W.cols] (see header docs).
// `act_scratch` is the persistent int8 activation staging buffer of the
// portable decode path (Gguf::quant_gemv_int8_scratch_bytes(K) bytes, see
// KokkosDNN_GGUFGemvInt8.hpp). Empty means "no staging" (the portable
// unit-dot kernel path).

template <class ExecSpace, class Scalar>
void linear_prefill(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Gguf::QuantWeight<ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "Inkling::linear_prefill") {
  if (staging.extent(0) != W.rows || staging.extent(1) != W.cols)
    throw std::invalid_argument("linear_prefill: staging shape mismatch in " +
                                label + " (staging [" +
                                std::to_string(staging.extent(0)) + "x" +
                                std::to_string(staging.extent(1)) +
                                "] vs weight [" + std::to_string(W.rows) +
                                "x" + std::to_string(W.cols) + "])");
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
    const std::string& label = "Inkling::linear_decode",
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch =
        Kokkos::View<uint8_t*, ExecSpace>()) {
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
  Gguf::quant_gemv<ExecSpace, Scalar>(handle, x1, W, y1, act_scratch, label);
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
    const std::string& label = "Inkling::linear_chunk",
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch =
        Kokkos::View<uint8_t*, ExecSpace>()) {
  if (X.extent(0) == 1) {
    linear_decode(handle, X, W, staging, Y, label, act_scratch);
  } else {
    linear_prefill(handle, X, W, staging, Y, label);
  }
}

// Shared-input decode helper: quantize the single input row X ([1, K])
// once into act_scratch for reuse across the linears sharing it. Returns
// false when the row cannot take the int8 path (caller runs the regular
// per-GEMV path instead, so a decline never changes results). This is the
// ticket producer; the consumers are linear_decode_reuse /
// linear_chunk_reuse below (stage-2-only via GgufMvq::gemv_int8_reuse_act).
template <class ExecSpace, class Scalar>
bool quantize_shared_act(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch) {
  if (X.extent(0) != 1) return false;
  if (act_scratch.extent(0) == 0) return false;
  Kokkos::View<Scalar*, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      x1(X.data(), X.extent(1));
  return ::KokkosDNN::Impl::GgufMvq::quantize_act_q8_row<ExecSpace, Scalar>(
      x1, act_scratch.data(), act_scratch.extent(0));
}

// Reuse-aware decode: when staged is true, act_scratch already holds
// quantize_shared_act(X) for this X/K and the dot runs stage-2-only
// (bit-identical to the per-GEMV path). Anything the reuse path declines
// (unsupported dtype/stride/align, undersized scratch) falls through to the
// regular per-GEMV decode, preserving the current decline rules. Dense
// weights ignore the ticket outright.

// Dense weights need no int8 staging; the ticket is ignored.
template <class ExecSpace, class Scalar>
void linear_decode_reuse(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch, bool staged,
    const std::string& label = "Inkling::linear_decode_reuse") {
  (void)act_scratch;
  (void)staged;
  linear_decode(handle, X, W, staging, Y, label);
}

template <class ExecSpace, class Scalar>
void linear_decode_reuse(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Gguf::QuantWeight<ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch, bool staged,
    const std::string& label = "Inkling::linear_decode_reuse") {
  if (X.extent(0) != 1 || Y.extent(0) != 1)
    throw std::invalid_argument("linear_decode_reuse: expected single-row X and Y");
  if (X.extent(1) != W.cols || Y.extent(1) != W.rows)
    throw std::invalid_argument("linear_decode_reuse: dimension mismatch");
  if (W.rows == 0 || W.cols == 0) return;
  if (staged) {
    ::KokkosDNN::Impl::GgufMvq::Wd mdt{};
    if (::KokkosDNN::Gguf::Impl::quant_int8_dtype(W.dtype, mdt)) {
      Kokkos::View<Scalar*, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          y1(Y.data(), W.rows);
      if (::KokkosDNN::Impl::GgufMvq::gemv_int8_reuse_act<ExecSpace, Scalar>(
              W.data.data(), mdt, W.rows, W.row_bytes(), act_scratch.data(),
              act_scratch.extent(0), W.cols, y1))
        return;
    }
    // Declined reuse: fall through to the per-GEMV decode below (which
    // re-quantizes on the int8 path or keeps the portable kernel).
  }
  linear_decode(handle, X, W, staging, Y, label, act_scratch);
}

// Chunk entry point with a staging ticket: decode reuses the shared
// quantization when staged, prefill (M > 1) takes the dense dequant+gemm
// path and ignores the ticket.
template <class ExecSpace, class Scalar, class Weight>
void linear_chunk_reuse(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Weight& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch, bool staged,
    const std::string& label = "Inkling::linear_chunk_reuse") {
  if (X.extent(0) == 1) {
    linear_decode_reuse(handle, X, W, staging, Y, act_scratch, staged, label);
  } else {
    (void)act_scratch;
    (void)staged;
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
    const std::string& label = "lm_head",
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch =
        Kokkos::View<uint8_t*, ExecSpace>()) {
  (void)act_scratch;  // dense weights need no int8 staging
  gemv<ExecSpace, Scalar>(handle, last, W, logits, label);
}

template <class ExecSpace, class Scalar>
void lm_head_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar*, ExecSpace>& last,
    const Gguf::QuantWeight<ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& logits,
    const std::string& label = "lm_head",
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch =
        Kokkos::View<uint8_t*, ExecSpace>()) {
  Gguf::quant_gemv<ExecSpace, Scalar>(handle, last, W, logits, act_scratch,
                                      label);
}

// ---- Embedding gather (dense table vs packed-row dequant) -----------------

template <class ExecSpace, class Scalar>
void embed_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table,
    const Kokkos::View<int*, ExecSpace>& tokens,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "Inkling::embed") {
  embedding_gather<ExecSpace, Scalar>(table, tokens, out, label);
}

// Packed-row gather: out[t,d] = dequant(table[token[t]], d) via the shared
// device element decoder (same values as dequant_to_dense, FP32 -> Scalar).
// One row is read per token (no M-times redundancy as in quant_gemm), so
// on-the-fly decode is the right trade here; no staging needed. Token ids
// must be in range, exactly as with the dense embedding_gather above.
template <class ExecSpace, class Scalar>
void embed_forward(
    const Gguf::QuantWeight<ExecSpace>& table,
    const Kokkos::View<int*, ExecSpace>& tokens,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "Inkling::embed_quant") {
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
}  // namespace Inkling
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_INKLING_QUANT_DISPATCH_HPP
