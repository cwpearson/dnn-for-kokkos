#ifndef KOKKOSDNN_MOSSIMAGEENCODER_HPP
#define KOKKOSDNN_MOSSIMAGEENCODER_HPP

// Qwen3.5 text-encoder path of OpenMOSS-Team/MOSS-Image-SFT (see
// QWEN_NOTES.md for the verified ground truth and examples/moss_image/
// MOSS_ARCH_INVESTIGATION.md section 4 for the pipeline context).
//
// Verified against transformers 5.12.1
// (transformers/models/qwen3_5/modeling_qwen3_5.py, reference/ mirror) and
// text_encoder/config.json @ main. All layer math below — including the
// Gated-DeltaNet linear attention — is pinned to the transformers torch
// fallbacks; there are no stubs.
//
// Model: hidden 4096, 32 layers, 16 Q / 4 KV heads x head_dim 256,
// intermediate 12288, SiLU SwiGLU MLP (no biases), attn_output_gate, hybrid
// 3x linear_attention + 1x full_attention (full at layers 3,7,11,15,19,23,
// 27,31), linear-attn key 16x128 / value 32x128, depthwise causal conv k=4
// + SiLU, Gated DeltaNet recurrence in fp32, partial rotary 0.25 with
// theta 10M and interleaved mrope [11,11,10] (no-op for text-only input),
// RMSNorm eps 1e-6 ((1+w) variant, zero-init; gated variant is plain-w),
// vocab 248320. encode_prompt: per-prompt unpadded forwards,
// chat_template enable_thinking=True, truncation at max_length, zero-padded
// batch + bool mask. This header consumes token ids (NOT raw text);
// tokenization/BPE stays host-side in scripts/moss_encode_qwen.py.
//
// TPL mapping (no hand-written cublas/cudnn calls anywhere here):
//   * Every linear goes through KokkosDNN::gemm (alpha=1, beta=0), which
//     owns the cuBLAS / hipBLAS ladder with a portable tiled fallback.
//     All Qwen linears are unbiased, so gemm (not gemm_bias) is exact.
//   * Full-attention layers go through attention_gqa_single (causal GQA,
//     matching the reference create_causal_mask prefill with q_off=0),
//     which owns the cuDNN SDPA ladder with a native streaming fallback.
//     NOTE: bidirectional attention is deliberately NOT used: the reference
//     text encoder is a causal decoder-only LM.
//   * embedding_gather, depthwise_conv1d_causal_forward, silu_forward,
//     swiglu are reused core ops (native elementwise; no TPL expresses
//     them). Everything else in Qwen35::Impl is a native portable
//     elementwise kernel (MDRange/RangePolicy only).
//
// Templated <ExecSpace, Scalar> (float and bhalf_t), FP32 accumulate.
// Internals in KokkosDNN::Qwen35::Impl. No backend #ifdefs.

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Conv1D.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_SwiGLU.hpp"

namespace KokkosDNN {
namespace Qwen35 {

// Model / encoder constants for the MOSS-Image-SFT text encoder
// (Qwen3_5Model text path; vision tower not executed by encode_prompt).
struct Config {
  int vocab_size = 248320;
  int hidden = 4096;
  int num_layers = 32;
  int num_q_heads = 16;
  int num_kv_heads = 4;
  int head_dim = 256;
  int intermediate = 12288;
  int lin_k_heads = 16;
  int lin_k_dim = 128;
  int lin_v_heads = 32;
  int lin_v_dim = 128;
  int conv_k = 4;
  float partial_rotary = 0.25f;
  double rope_theta = 10000000.0;
  int mrope_section[3] = {11, 11, 10};
  float rms_eps = 1e-6f;
  bool attn_output_gate = true;
  bool attn_bias = false;  // q/k/v/o + MLP + in/out projs all unbiased

  int q_dim() const { return num_q_heads * head_dim; }
  int kv_dim() const { return num_kv_heads * head_dim; }
  int lin_key_dim() const { return lin_k_heads * lin_k_dim; }
  int lin_value_dim() const { return lin_v_heads * lin_v_dim; }
  int lin_qkv_dim() const { return 2 * lin_key_dim() + lin_value_dim(); }
  int rotary_dim() const {
    return static_cast<int>(static_cast<float>(head_dim) * partial_rotary);
  }
  // Full attention at 0-based layers 3,7,...,31 (1-based multiples of the
  // full_attention_interval=4); all others linear_attention.
  bool is_full_layer(int l) const { return (l + 1) % 4 == 0; }

  static Config moss_image_sft() { return Config{}; }

  // Tiny config for unit tests (4 layers: 0,1,2 linear + 3 full).
  static Config tiny_test() {
    Config c;
    c.vocab_size = 64;
    c.hidden = 32;
    c.num_layers = 4;
    c.num_q_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 8;
    c.intermediate = 64;
    c.lin_k_heads = 2;
    c.lin_k_dim = 8;
    c.lin_v_heads = 4;
    c.lin_v_dim = 8;
    c.conv_k = 4;
    c.rope_theta = 10000.0;
    return c;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Qwen35::Config::validate: " + m);
    };
    if (vocab_size <= 0) bad("vocab_size must be > 0");
    if (hidden <= 0) bad("hidden must be > 0");
    if (num_layers <= 0) bad("num_layers must be > 0");
    if (num_q_heads <= 0 || num_kv_heads <= 0) bad("head counts must be > 0");
    if (num_q_heads % num_kv_heads != 0)
      bad("num_q_heads must be a multiple of num_kv_heads");
    if (head_dim <= 0) bad("head_dim must be > 0");
    if (intermediate <= 0) bad("intermediate must be > 0");
    if (lin_k_heads <= 0 || lin_k_dim <= 0 || lin_v_heads <= 0 ||
        lin_v_dim <= 0)
      bad("linear-attention dims must be > 0");
    if (lin_v_heads % lin_k_heads != 0)
      bad("lin_v_heads must be a multiple of lin_k_heads");
    if (conv_k < 1) bad("conv_k must be >= 1");
    if (!(partial_rotary > 0.0f && partial_rotary <= 1.0f))
      bad("partial_rotary must be in (0, 1]");
    if (rotary_dim() <= 0 || rotary_dim() % 2 != 0 || rotary_dim() > head_dim)
      bad("rotary_dim must be a positive even number <= head_dim");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (!(rms_eps >= 0.0f) || !std::isfinite(rms_eps))
      bad("rms_eps must be finite and nonnegative");
    if (!attn_output_gate) bad("attn_output_gate must be true (checkpoint)");
    if (attn_bias) bad("attn_bias must be false (checkpoint)");
  }
};

namespace Impl {

// -- FP32 scalars (device-callable) ---------------------------------------
KOKKOS_INLINE_FUNCTION float qwen_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
KOKKOS_INLINE_FUNCTION float qwen_softplus(float x) {
  return x > 20.0f ? x : logf(1.0f + expf(x));
}
KOKKOS_INLINE_FUNCTION float qwen_silu(float x) {
  return x * qwen_sigmoid(x);
}
KOKKOS_INLINE_FUNCTION float qwen_l2inv(float ss, int n, float eps) {
  return 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
}

// FLA sum-based L2 norm (NOT an RMSNorm): 1/sqrt(sum + eps), matching the
// transformers l2norm() used with use_qk_l2norm_in_kernel=True. Deliberately
// different from the mean-based qwen_l2inv above.

// bfloat16 -> float (host; safetensors BF16 payloads).
inline float bf16_to_float(uint16_t b) {
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &x, sizeof(float));
  return f;
}
// -- (1+w) per-head QK RMSNorm ---------------------------------------------
// Qwen3_5RMSNorm applied per attention head (full-layer q_norm/k_norm):
//   out[t,h,d] = x[t,h,:]/rms(x[t,h,:],eps) * (1 + w[d]).
// The core qk_rmsnorm3d op uses a plain multiply, so this local kernel pins
// the (1+w) variant. In-place safe. Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar, class XView, class WView,
          class YView>
void qk_rmsnorm_oneplus_3d(const XView& x, const WView& w, float eps,
                           const YView& y,
                           const std::string& label = "Qwen35::qk_norm") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (y.extent(0) != T || y.extent(1) != H || y.extent(2) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("qk_rmsnorm_oneplus_3d: dim mismatch");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("qk_rmsnorm_oneplus_3d: bad eps");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(t, h, d));
          ss += v * v;
        }
        float inv = qwen_l2inv(ss, static_cast<int>(D), eps);
        for (size_t d = 0; d < D; ++d)
          y(t, h, d) = static_cast<Scalar>(
              static_cast<float>(x(t, h, d)) * inv *
              (1.0f + static_cast<float>(w(d))));
      });
}

// -- Qwen partial rotary apply (in-place) -----------------------------------
// Reference: modeling_qwen3_5.apply_rotary_pos_emb with rotary_dim = R
// (first R dims), rotate_half pairs (d, d+R/2), angles pos*inv[j],
// inv[j] = theta^(-2j/R) (denominator is R, verified against
// compute_default_rope_parameters). For text-only encoder forwards the
// mrope interleave is an exact no-op, so positions are 0..T-1.
// x: [T, H, D] with R <= D even. Dims [R, D) pass through.
// In-place safe. Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar, class XView, class FView>
void qwen_partial_rope_inplace(const XView& x, const FView& inv_freq,
                               int rotary_dim,
                               const std::string& label = "Qwen35::rope") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  const size_t R = static_cast<size_t>(rotary_dim);
  const size_t Rh = R / 2;
  if (R == 0 || R % 2 != 0 || R > D)
    throw std::invalid_argument("qwen_partial_rope_inplace: bad rotary_dim");
  if (inv_freq.extent(0) != Rh)
    throw std::invalid_argument("qwen_partial_rope_inplace: inv_freq dim");
  if (T == 0 || H == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const float pos = static_cast<float>(t);
        for (size_t j = 0; j < Rh; ++j) {
          const float ang = pos * inv_freq(j);
          const float c = cosf(ang), s = sinf(ang);
          const float a = static_cast<float>(x(t, h, j));
          const float b = static_cast<float>(x(t, h, j + Rh));
          x(t, h, j) = static_cast<Scalar>(a * c - b * s);
          x(t, h, j + Rh) = static_cast<Scalar>(a * s + b * c);
        }
      });
}

// Host mirror of the inv_freq table: inv[j] = theta^(-2j/R), j < R/2.
inline std::vector<float> qwen_rope_inv_host(int rotary_dim, double theta) {
  const size_t Rh = static_cast<size_t>(rotary_dim) / 2;
  std::vector<float> out(Rh);
  const double R = static_cast<double>(rotary_dim);
  for (size_t j = 0; j < Rh; ++j)
    out[j] = static_cast<float>(1.0 / std::pow(theta, (2.0 * (double)j) / R));
  return out;
}

// -- Output gating -----------------------------------------------------------
// Full-layer attn_output_gate: out[t,i] = attn[t,i] * sigmoid(gate[t,i]).
// Same-index math; out may alias attn and/or gate.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar, class AView, class GView,
          class OView>
void qwen_gate_multiply(const AView& attn, const GView& gate, const OView& out,
                        const std::string& label = "Qwen35::gate_mul") {
  const size_t T = attn.extent(0), N = attn.extent(1);
  if (gate.extent(0) != T || gate.extent(1) != N || out.extent(0) != T ||
      out.extent(1) != N)
    throw std::invalid_argument("qwen_gate_multiply: dim mismatch");
  if (T == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, N}),
      KOKKOS_LAMBDA(size_t t, size_t i) {
        out(t, i) = static_cast<Scalar>(
            static_cast<float>(attn(t, i)) *
            qwen_sigmoid(static_cast<float>(gate(t, i))));
      });
}

// -- Residual add (in-place on x) --------------------------------------------
template <class ExecSpace, class Scalar, class XView, class BView>
void qwen_residual_add(const XView& x, const BView& branch,
                       const std::string& label = "Qwen35::residual") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (branch.extent(0) != T || branch.extent(1) != D)
    throw std::invalid_argument("qwen_residual_add: dim mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        x(t, d) = static_cast<Scalar>(static_cast<float>(x(t, d)) +
                                      static_cast<float>(branch(t, d)));
      });
}
// -- Linear-layer gate precompute ---------------------------------------------
// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), fp32.
// a, b: [T, Hv]; a_log, dt: [Hv]; beta, g: [T, Hv] float.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar, class AView, class BView,
          class PView, class OView>
void qwen_gdn_gates(const AView& a, const BView& b, const PView& a_log,
                    const PView& dt, const OView& beta, const OView& g,
                    const std::string& label = "Qwen35::gdn_gates") {
  const size_t T = a.extent(0), H = a.extent(1);
  if (b.extent(0) != T || b.extent(1) != H || a_log.extent(0) != H ||
      dt.extent(0) != H || beta.extent(0) != T || beta.extent(1) != H ||
      g.extent(0) != T || g.extent(1) != H)
    throw std::invalid_argument("qwen_gdn_gates: dim mismatch");
  if (T == 0 || H == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const float av = static_cast<float>(a(t, h));
        const float bv = static_cast<float>(b(t, h));
        const float A = expf(static_cast<float>(a_log(h)));
        beta(t, h) = qwen_sigmoid(bv);
        g(t, h) = -A * qwen_softplus(av + static_cast<float>(dt(h)));
      });
}

// -- Head repeat (linear-layer q/k lin_k_heads -> lin_v_heads) -----------------
// out[t, h*rep + r, d] = x[t, h, d] (repeat_interleave).
// Dispatch: native data movement (no TPL).
template <class ExecSpace, class Scalar, class XView, class YView>
void qwen_head_repeat(const XView& x, const YView& y, int rep,
                      const std::string& label = "Qwen35::head_repeat") {
  const size_t T = x.extent(0), Hi = x.extent(1), D = x.extent(2);
  if (rep < 1) throw std::invalid_argument("qwen_head_repeat: bad rep");
  const size_t Ho = Hi * static_cast<size_t>(rep);
  if (y.extent(0) != T || y.extent(1) != Ho || y.extent(2) != D)
    throw std::invalid_argument("qwen_head_repeat: dim mismatch");
  if (T == 0 || Hi == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {T, Ho, D}),
      KOKKOS_LAMBDA(size_t t, size_t ho, size_t d) {
        y(t, ho, d) = x(t, ho / static_cast<size_t>(rep), d);
      });
}

// -- QK L2-norm (+ optional scale for q) ----------------------------------------
// FLA l2norm(x, eps) per (t,h) row: x/sqrt(sum(x^2) + eps) (sum, NOT mean),
// then q is additionally scaled by 1/sqrt(Kdim) by passing scale != 1.
// In-place safe. Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar, class XView>
void qwen_qk_l2norm(const XView& x, float eps, float scale,
                    const std::string& label = "Qwen35::qk_l2norm") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("qwen_qk_l2norm: bad eps");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(t, h, d));
          ss += v * v;
        }
        float inv = (1.0f / sqrtf(ss + eps)) * scale;
        for (size_t d = 0; d < D; ++d)
          x(t, h, d) =
              static_cast<Scalar>(static_cast<float>(x(t, h, d)) * inv);
      });
}
// -- Recurrent gated delta rule (linear-layer core) -----------------------------
// Exact port of torch_recurrent_gated_delta_rule: per head h with state
// S [K, V] (fp32 workspace, zeroed by caller):
//   S *= exp(g); kv = sum_k S[k,v] k[k]; delta = (v - kv) beta;
//   S += k (x) delta; out = sum_k S[k,v] q[k].
// q/k are L2-normed with q pre-scaled by 1/sqrt(K); beta/g fp32 [T, H].
// Parallel over heads; T is sequential per head (recurrence dependence).
// Dispatch: native (no TPL; a chunked parallel form is future work).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class GView, class SView, class OView>
void qwen_gated_delta_recurrent(const QView& q, const KView& k, const VView& v,
                                const GView& beta, const GView& g,
                                const SView& state, const OView& out,
                                const std::string& label = "Qwen35::gdn") {
  const size_t T = q.extent(0), H = q.extent(1), K = q.extent(2);
  const size_t V = v.extent(2);
  if (k.extent(0) != T || k.extent(1) != H || k.extent(2) != K ||
      v.extent(0) != T || v.extent(1) != H || beta.extent(0) != T ||
      beta.extent(1) != H || g.extent(0) != T || g.extent(1) != H ||
      state.extent(0) != H || state.extent(1) != K || state.extent(2) != V ||
      out.extent(0) != T || out.extent(1) != H || out.extent(2) != V)
    throw std::invalid_argument("qwen_gated_delta_recurrent: dim mismatch");
  if (T == 0 || H == 0 || K == 0 || V == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, H),
      KOKKOS_LAMBDA(size_t h) {
        for (size_t t = 0; t < T; ++t) {
          const float decay = expf(g(t, h));
          const float bt = beta(t, h);
          for (size_t kk = 0; kk < K; ++kk)
            for (size_t vv = 0; vv < V; ++vv) state(h, kk, vv) *= decay;
          for (size_t vv = 0; vv < V; ++vv) {
            float kv = 0;
            for (size_t kk = 0; kk < K; ++kk)
              kv += state(h, kk, vv) * static_cast<float>(k(t, h, kk));
            const float delta =
                (static_cast<float>(v(t, h, vv)) - kv) * bt;
            for (size_t kk = 0; kk < K; ++kk)
              state(h, kk, vv) += static_cast<float>(k(t, h, kk)) * delta;
          }
          for (size_t vv = 0; vv < V; ++vv) {
            float o = 0;
            for (size_t kk = 0; kk < K; ++kk)
              o += state(h, kk, vv) * static_cast<float>(q(t, h, kk));
            out(t, h, vv) = static_cast<Scalar>(o);
          }
        }
      });
}

// -- Gated RMSNorm (linear-layer norm) -------------------------------------------
// Qwen3_5RMSNormGated: out = rms(core,eps) * w * silu(z), fp32.
// core/z/out: [T, H, D]; w: [D] (ones-init, plain multiply).
// Same-index math; out may alias core and/or z.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar, class CView, class ZView, class WView,
          class OView>
void qwen_rmsnorm_gated(const CView& core, const ZView& z, const WView& w,
                        float eps, const OView& out,
                        const std::string& label = "Qwen35::rms_gated") {
  const size_t T = core.extent(0), H = core.extent(1), D = core.extent(2);
  if (z.extent(0) != T || z.extent(1) != H || z.extent(2) != D ||
      out.extent(0) != T || out.extent(1) != H || out.extent(2) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("qwen_rmsnorm_gated: dim mismatch");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("qwen_rmsnorm_gated: bad eps");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float c = static_cast<float>(core(t, h, d));
          ss += c * c;
        }
        float inv = qwen_l2inv(ss, static_cast<int>(D), eps);
        for (size_t d = 0; d < D; ++d) {
          const float n = static_cast<float>(core(t, h, d)) * inv *
                          static_cast<float>(w(d));
          out(t, h, d) = static_cast<Scalar>(
              n * qwen_silu(static_cast<float>(z(t, h, d))));
        }
      });
}
// -- Minimal safetensors reader ----------------------------------------------
// Reads the per-file header (u64 LE length + JSON) and builds a
// name -> {dtype, shape, data_begin} map for exactly the producer schema:
// {"name": {"dtype": "F32"|"BF16", "shape": [...],
// "data_offsets": [begin, end]}, ...} plus __metadata__.
// Only F32/BF16 payloads are accepted; anything else throws.
// Little-endian host required (checked at load).
struct SafeTensorInfo {
  std::string dtype;           // "F32" or "BF16"
  std::vector<int64_t> shape;  // row-major extents
  uint64_t data_begin = 0;     // absolute file offset of payload
  uint64_t nbytes = 0;
};

// Tiny JSON value tree (objects/arrays/strings/numbers only).
struct JsonVal {
  enum class T { Null, Num, Str, Arr, Obj } type = T::Null;
  double num = 0;
  std::string str;
  std::vector<JsonVal> arr;
  std::map<std::string, JsonVal> obj;
};

struct JsonParser {
  const char* p = nullptr;
  const char* end = nullptr;
  [[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("safetensors header JSON: " + m);
  }
  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
      ++p;
  }
  char peek() {
    if (p >= end) fail("unexpected end");
    return *p;
  }
  void expect(const char* lit) {
    for (const char* q = lit; *q; ++q, ++p)
      if (p >= end || *p != *q) fail("bad literal");
  }
  JsonVal parse() {
    ws();
    JsonVal v = parse_val();
    ws();
    if (p != end) fail("trailing characters");
    return v;
  }
  JsonVal parse_val() {
    ws();
    char c = peek();
    if (c == '{') return parse_obj();
    if (c == '[') return parse_arr();
    if (c == '"') {
      JsonVal v;
      v.type = JsonVal::T::Str;
      v.str = parse_str();
      return v;
    }
    if (c == 'n') {
      expect("null");
      return JsonVal{};
    }
    if (c == 't') {
      expect("true");
      JsonVal v;
      v.type = JsonVal::T::Num;
      v.num = 1;
      return v;
    }
    if (c == 'f') {
      expect("false");
      JsonVal v;
      v.type = JsonVal::T::Num;
      v.num = 0;
      return v;
    }
    return parse_num();
  }
  JsonVal parse_obj() {
    JsonVal v;
    v.type = JsonVal::T::Obj;
    ++p;  // {
    ws();
    if (peek() == '}') {
      ++p;
      return v;
    }
    while (true) {
      ws();
      if (peek() != '"') fail("object key must be a string");
      std::string k = parse_str();
      ws();
      if (peek() != ':') fail("expected ':'");
      ++p;
      v.obj[k] = parse_val();
      ws();
      char c = peek();
      if (c == ',') {
        ++p;
        continue;
      }
      if (c == '}') {
        ++p;
        break;
      }
      fail("expected ',' or '}'");
    }
    return v;
  }
  JsonVal parse_arr() {
    JsonVal v;
    v.type = JsonVal::T::Arr;
    ++p;  // [
    ws();
    if (peek() == ']') {
      ++p;
      return v;
    }
    while (true) {
      v.arr.push_back(parse_val());
      ws();
      char c = peek();
      if (c == ',') {
        ++p;
        continue;
      }
      if (c == ']') {
        ++p;
        break;
      }
      fail("expected ',' or ']'");
    }
    return v;
  }
  std::string parse_str() {
    ++p;  // opening quote
    std::string s;
    while (true) {
      if (p >= end) fail("unterminated string");
      char c = *p++;
      if (c == '"') break;
      if (c == '\\') {
        if (p >= end) fail("bad escape");
        char e = *p++;
        switch (e) {
          case '"': s += '"'; break;
          case '\\': s += '\\'; break;
          case '/': s += '/'; break;
          case 'n': s += '\n'; break;
          case 't': s += '\t'; break;
          case 'r': s += '\r'; break;
          case 'u': {
            if (end - p < 4) fail("bad \\u escape");
            unsigned cp = 0;
            for (int i = 0; i < 4; ++i) {
              char h = *p++;
              cp <<= 4;
              if (h >= '0' && h <= '9') cp += (unsigned)(h - '0');
              else if (h >= 'a' && h <= 'f') cp += (unsigned)(h - 'a') + 10;
              else if (h >= 'A' && h <= 'F') cp += (unsigned)(h - 'A') + 10;
              else fail("bad \\u escape");
            }
            // Encode BMP code point as UTF-8 (headers are ASCII in
            // practice; lone surrogates are replaced).
            if (cp < 0x80) {
              s += (char)cp;
            } else if (cp < 0x800) {
              s += (char)(0xC0 | (cp >> 6));
              s += (char)(0x80 | (cp & 0x3F));
            } else {
              s += (char)(0xE0 | (cp >> 12));
              s += (char)(0x80 | ((cp >> 6) & 0x3F));
              s += (char)(0x80 | (cp & 0x3F));
            }
            break;
          }
          default: fail("bad escape");
        }
      } else {
        s += c;
      }
    }
    return s;
  }
  JsonVal parse_num() {
    const char* b = p;
    if (p < end && (*p == '-' || *p == '+')) ++p;
    bool any = false;
    while (p < end &&
           ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' ||
            *p == 'E' || *p == '-' || *p == '+')) {
      ++p;
      any = true;
    }
    if (!any) fail("bad number");
    JsonVal v;
    v.type = JsonVal::T::Num;
    try {
      v.num = std::stod(std::string(b, p));
    } catch (...) {
      fail("bad number");
    }
    return v;
  }
};

// Parse one .safetensors header; returns name map + total header size
// (8 + json_len) so callers can seek to payloads. Throws on I/O or
// schema errors.
inline std::map<std::string, SafeTensorInfo> read_safetensors_header(
    const std::string& path, uint64_t& header_size) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open safetensors file: " + path);
  uint64_t json_len = 0;
  f.read(reinterpret_cast<char*>(&json_len), 8);
  if (!f) throw std::runtime_error("cannot read safetensors header: " + path);
  if (json_len > (uint64_t{1} << 32))
    throw std::runtime_error("safetensors header implausibly large: " + path);
  std::string js((size_t)json_len, '\0');
  f.read(js.data(), (std::streamsize)json_len);
  if (!f) throw std::runtime_error("cannot read safetensors JSON: " + path);
  header_size = 8 + json_len;
  JsonParser ps{js.data(), js.data() + js.size()};
  JsonVal root = ps.parse();
  if (root.type != JsonVal::T::Obj) throw std::runtime_error("bad header");
  std::map<std::string, SafeTensorInfo> out;
  for (const auto& kv : root.obj) {
    if (kv.first == "__metadata__") continue;
    const JsonVal& e = kv.second;
    if (e.type != JsonVal::T::Obj) throw std::runtime_error("bad entry");
    auto it_d = e.obj.find("dtype");
    auto it_s = e.obj.find("shape");
    auto it_o = e.obj.find("data_offsets");
    if (it_d == e.obj.end() || it_s == e.obj.end() || it_o == e.obj.end())
      throw std::runtime_error("entry missing keys: " + kv.first);
    if (it_d->second.type != JsonVal::T::Str)
      throw std::runtime_error("bad dtype: " + kv.first);
    SafeTensorInfo info;
    info.dtype = it_d->second.str;
    if (info.dtype != "F32" && info.dtype != "BF16")
      throw std::runtime_error("unsupported dtype '" + info.dtype +
                               "' for tensor " + kv.first +
                               " (only F32/BF16)");
    if (it_s->second.type != JsonVal::T::Arr)
      throw std::runtime_error("bad shape: " + kv.first);
    for (const auto& d : it_s->second.arr) {
      if (d.type != JsonVal::T::Num || d.num < 0)
        throw std::runtime_error("bad shape dim: " + kv.first);
      info.shape.push_back((int64_t)d.num);
    }
    if (it_o->second.type != JsonVal::T::Arr ||
        it_o->second.arr.size() != 2 ||
        it_o->second.arr[0].type != JsonVal::T::Num ||
        it_o->second.arr[1].type != JsonVal::T::Num)
      throw std::runtime_error("bad data_offsets: " + kv.first);
    uint64_t b = (uint64_t)it_o->second.arr[0].num;
    uint64_t en = (uint64_t)it_o->second.arr[1].num;
    if (en < b) throw std::runtime_error("bad offsets: " + kv.first);
    info.data_begin = header_size + b;
    info.nbytes = en - b;
    out[kv.first] = info;
  }
  return out;
}
// Convert one raw payload (F32/BF16, row-major, contiguous) into a fresh
// host float buffer. Validates byte size against numel.
inline void convert_payload_to_float(const std::string& name,
                                     const SafeTensorInfo& info,
                                     const char* raw, int64_t numel,
                                     std::vector<float>& dst) {
  int64_t have = 0;
  if (info.dtype == "F32") {
    have = (int64_t)(info.nbytes / 4);
    if (have != numel)
      throw std::runtime_error("tensor " + name + ": byte size mismatch");
    dst.resize((size_t)numel);
    std::memcpy(dst.data(), raw, (size_t)numel * 4);
  } else if (info.dtype == "BF16") {
    have = (int64_t)(info.nbytes / 2);
    if (have != numel)
      throw std::runtime_error("tensor " + name + ": byte size mismatch");
    dst.resize((size_t)numel);
    const uint16_t* u = reinterpret_cast<const uint16_t*>(raw);
    for (int64_t i = 0; i < numel; ++i) dst[(size_t)i] = bf16_to_float(u[i]);
  } else {
    throw std::runtime_error("tensor " + name + ": unsupported dtype");
  }
}

inline void require_shape(const std::string& name,
                          const SafeTensorInfo& info,
                          const std::vector<int64_t>& want) {
  if (info.shape != want) {
    std::string m = "tensor " + name + ": shape mismatch (file [";
    for (size_t i = 0; i < info.shape.size(); ++i)
      m += (i ? "," : "") + std::to_string(info.shape[i]);
    m += "] vs want [";
    for (size_t i = 0; i < want.size(); ++i)
      m += (i ? "," : "") + std::to_string(want[i]);
    m += "])";
    throw std::runtime_error(m);
  }
}

// Fill one device view from a host float buffer (casts to Scalar).
// Exactly one deep_copy per call. Rank-1 and rank-2 overloads.
template <class ExecSpace, class Scalar>
void fill_from_float(
    const Kokkos::View<Scalar*, ExecSpace>& dst,
    const std::vector<float>& src, const std::string& name) {
  if (dst.extent(0) != src.size())
    throw std::runtime_error("tensor " + name + ": size mismatch");
  auto hm = Kokkos::create_mirror_view(
      typename Kokkos::View<Scalar*, ExecSpace>::host_mirror_space(), dst);
  for (size_t i = 0; i < src.size(); ++i)
    hm(i) = static_cast<Scalar>(src[i]);
  Kokkos::deep_copy(dst, hm);
}

template <class ExecSpace, class Scalar>
void fill_from_float(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst,
    const std::vector<float>& src, const std::string& name) {
  if (dst.extent(0) * dst.extent(1) != src.size())
    throw std::runtime_error("tensor " + name + ": size mismatch");
  auto hm = Kokkos::create_mirror_view(
      typename Kokkos::View<Scalar**, Kokkos::LayoutRight,
                            ExecSpace>::host_mirror_space(),
      dst);
  size_t k = 0;
  for (size_t i = 0; i < dst.extent(0); ++i)
    for (size_t j = 0; j < dst.extent(1); ++j) hm(i, j) = static_cast<Scalar>(src[k++]);
  Kokkos::deep_copy(dst, hm);
}

}  // namespace Impl
// Device-resident Qwen3.5 text-encoder weights (language_model.* only;
// visual.* tower is never executed by encode_prompt and is not loaded).
// Layouts mirror the safetensors row-major shapes: Linear [out, in]
// loads directly into gemm's W [N, K]; norms/vectors are rank-1.
template <class ExecSpace, class Scalar>
struct Weights {
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;

  struct FullLayer {
    View2 q_proj;  // [2*Nq*Hd, H] (query || gate chunk)
    View2 k_proj;  // [Nkv*Hd, H]
    View2 v_proj;  // [Nkv*Hd, H]
    View2 o_proj;  // [H, Nq*Hd]
    View1 q_norm;  // [Hd] ((1+w) RMSNorm)
    View1 k_norm;  // [Hd]
  };
  struct LinearLayer {
    View2 in_qkv;  // [2*Kdim+Vdim, H]
    View2 in_z;    // [Vdim, H]
    View2 in_a;    // [Hv, H]
    View2 in_b;    // [Hv, H]
    View2 conv;    // [C=2*Kdim+Vdim, K] (squeezed from [C,1,K])
    View2 out_proj;  // [H, Vdim]
    View1 norm;      // [Vd] (gated RMSNorm, plain multiply)
    View1 a_log;     // [Hv] float math (stored as Scalar, used as float)
    View1 dt_bias;   // [Hv]
  };
  struct Layer {
    bool is_full = false;
    View1 input_norm;   // [H] ((1+w) RMSNorm)
    View1 post_norm;    // [H]
    View2 gate_proj;    // [I, H]
    View2 up_proj;      // [I, H]
    View2 down_proj;    // [H, I]
    FullLayer full;
    LinearLayer lin;
  };

  Config cfg;
  View2 embed;  // [V, H]
  View1 final_norm;  // [H]
  std::vector<Layer> layers;

  Weights() = default;
  explicit Weights(const Config& c) : cfg(c) {
    cfg.validate();
    embed = View2("Qwen35::embed", cfg.vocab_size, cfg.hidden);
    final_norm = View1("Qwen35::final_norm", cfg.hidden);
    layers.resize((size_t)cfg.num_layers);
    for (int l = 0; l < cfg.num_layers; ++l) {
      Layer& L = layers[(size_t)l];
      L.is_full = cfg.is_full_layer(l);
      L.input_norm = View1("Qwen35::in_norm", cfg.hidden);
      L.post_norm = View1("Qwen35::post_norm", cfg.hidden);
      L.gate_proj = View2("Qwen35::gate", cfg.intermediate, cfg.hidden);
      L.up_proj = View2("Qwen35::up", cfg.intermediate, cfg.hidden);
      L.down_proj = View2("Qwen35::down", cfg.hidden, cfg.intermediate);
      if (L.is_full) {
        L.full.q_proj = View2("Qwen35::q", 2 * cfg.q_dim(), cfg.hidden);
        L.full.k_proj = View2("Qwen35::k", cfg.kv_dim(), cfg.hidden);
        L.full.v_proj = View2("Qwen35::v", cfg.kv_dim(), cfg.hidden);
        L.full.o_proj = View2("Qwen35::o", cfg.hidden, cfg.q_dim());
        L.full.q_norm = View1("Qwen35::qn", cfg.head_dim);
        L.full.k_norm = View1("Qwen35::kn", cfg.head_dim);
      } else {
        L.lin.in_qkv = View2("Qwen35::qkv", cfg.lin_qkv_dim(), cfg.hidden);
        L.lin.in_z = View2("Qwen35::z", cfg.lin_value_dim(), cfg.hidden);
        L.lin.in_a = View2("Qwen35::a", cfg.lin_v_heads, cfg.hidden);
        L.lin.in_b = View2("Qwen35::b", cfg.lin_v_heads, cfg.hidden);
        L.lin.conv = View2("Qwen35::conv", cfg.lin_qkv_dim(), cfg.conv_k);
        L.lin.out_proj = View2("Qwen35::op", cfg.hidden, cfg.lin_value_dim());
        L.lin.norm = View1("Qwen35::ln", cfg.lin_v_dim);
        L.lin.a_log = View1("Qwen35::Alog", cfg.lin_v_heads);
        L.lin.dt_bias = View1("Qwen35::dt", cfg.lin_v_heads);
      }
    }
  }

  void release() { *this = Weights(); }

  size_t resident_bytes() const {
    size_t n = 0;
    auto acc1 = [&n](const View1& v) { n += v.extent(0) * sizeof(Scalar); };
    auto acc2 = [&n](const View2& v) {
      n += v.extent(0) * v.extent(1) * sizeof(Scalar);
    };
    acc2(embed);
    acc1(final_norm);
    for (const auto& L : layers) {
      acc1(L.input_norm);
      acc1(L.post_norm);
      acc2(L.gate_proj);
      acc2(L.up_proj);
      acc2(L.down_proj);
      if (L.is_full) {
        acc2(L.full.q_proj);
        acc2(L.full.k_proj);
        acc2(L.full.v_proj);
        acc2(L.full.o_proj);
        acc1(L.full.q_norm);
        acc1(L.full.k_norm);
      } else {
        acc2(L.lin.in_qkv);
        acc2(L.lin.in_z);
        acc2(L.lin.in_a);
        acc2(L.lin.in_b);
        acc2(L.lin.conv);
        acc2(L.lin.out_proj);
        acc1(L.lin.norm);
        acc1(L.lin.a_log);
        acc1(L.lin.dt_bias);
      }
    }
    return n;
  }

  // Deterministic pseudo-random init (uniform [-0.02, 0.02], the config
  // initializer_range) for offline tests without a checkpoint.
  void random_init(unsigned seed = 1234) {
    if (layers.empty()) throw std::logic_error("Weights: not allocated");
    uint64_t s = seed;
    auto rnd = [&s]() {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      return (float)((s >> 33) * (1.0 / 4294967296.0) - 0.5) * 0.04f;
    };
    auto f1 = [&rnd](View1& v) {
      auto hm = Kokkos::create_mirror_view(
          typename View1::host_mirror_space(), v);
      for (size_t i = 0; i < v.extent(0); ++i) hm(i) = Scalar(rnd());
      Kokkos::deep_copy(v, hm);
    };
    auto f2 = [&rnd](View2& v) {
      auto hm = Kokkos::create_mirror_view(
          typename View2::host_mirror_space(), v);
      for (size_t i = 0; i < v.extent(0); ++i)
        for (size_t j = 0; j < v.extent(1); ++j) hm(i, j) = Scalar(rnd());
      Kokkos::deep_copy(v, hm);
    };
    f2(embed);
    f1(final_norm);
    for (auto& L : layers) {
      f1(L.input_norm);
      f1(L.post_norm);
      f2(L.gate_proj);
      f2(L.up_proj);
      f2(L.down_proj);
      if (L.is_full) {
        f2(L.full.q_proj);
        f2(L.full.k_proj);
        f2(L.full.v_proj);
        f2(L.full.o_proj);
        f1(L.full.q_norm);
        f1(L.full.k_norm);
      } else {
        f2(L.lin.in_qkv);
        f2(L.lin.in_z);
        f2(L.lin.in_a);
        f2(L.lin.in_b);
        f2(L.lin.conv);
        f2(L.lin.out_proj);
        f1(L.lin.norm);
        f1(L.lin.a_log);
        f1(L.lin.dt_bias);
      }
    }
  }
};
// ---- checkpoint loading (defined after Weights) ---------------------------
// Streaming .safetensors loader: each required tensor is located via the
// per-file header map, read with one fseek+fread into a host buffer
// (never slurps whole multi-GB shards), converted F32/BF16 -> Scalar,
// and moved to device with exactly one deep_copy per tensor (see
// Impl::fill_from_float). Accepts all shards
// (model-0000{1..4}-of-00004.safetensors); visual.* tensors are ignored.
template <class ExecSpace, class Scalar>
void load_weights_safetensors(Weights<ExecSpace, Scalar>& w,
                              const std::vector<std::string>& files) {
  if (w.layers.empty()) throw std::logic_error("Weights: not allocated");
  if (files.empty())
    throw std::invalid_argument("load_weights_safetensors: no files");
  const Config& cfg = w.cfg;
  // Little-endian host required (safetensors payloads are LE).
  const uint16_t le_check = 0x0001;
  if (*reinterpret_cast<const uint8_t*>(&le_check) != 0x01)
    throw std::runtime_error("load_weights_safetensors: BE host");

  struct Need {
    std::string name;
    std::vector<int64_t> shape;  // empty => rank-3 conv special case
    int kind;  // 0: rank1, 1: rank2, 2: conv [C,1,K]->[C,K]
  };
  std::vector<Need> needs;
  needs.push_back({"language_model.embed_tokens.weight",
                   {(int64_t)cfg.vocab_size, (int64_t)cfg.hidden}, 1});
  needs.push_back(
      {"language_model.norm.weight", {(int64_t)cfg.hidden}, 0});
  for (int l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "language_model.layers." + std::to_string(l) + ".";
    needs.push_back({p + "input_layernorm.weight", {(int64_t)cfg.hidden}, 0});
    needs.push_back(
        {p + "post_attention_layernorm.weight", {(int64_t)cfg.hidden}, 0});
    needs.push_back({p + "mlp.gate_proj.weight",
                     {(int64_t)cfg.intermediate, (int64_t)cfg.hidden}, 1});
    needs.push_back({p + "mlp.up_proj.weight",
                     {(int64_t)cfg.intermediate, (int64_t)cfg.hidden}, 1});
    needs.push_back({p + "mlp.down_proj.weight",
                     {(int64_t)cfg.hidden, (int64_t)cfg.intermediate}, 1});
    if (cfg.is_full_layer(l)) {
      needs.push_back({p + "self_attn.q_proj.weight",
                       {(int64_t)2 * cfg.q_dim(), (int64_t)cfg.hidden}, 1});
      needs.push_back({p + "self_attn.k_proj.weight",
                       {(int64_t)cfg.kv_dim(), (int64_t)cfg.hidden}, 1});
      needs.push_back({p + "self_attn.v_proj.weight",
                       {(int64_t)cfg.kv_dim(), (int64_t)cfg.hidden}, 1});
      needs.push_back({p + "self_attn.o_proj.weight",
                       {(int64_t)cfg.hidden, (int64_t)cfg.q_dim()}, 1});
      needs.push_back({p + "self_attn.q_norm.weight",
                       {(int64_t)cfg.head_dim}, 0});
      needs.push_back({p + "self_attn.k_norm.weight",
                       {(int64_t)cfg.head_dim}, 0});
    } else {
      needs.push_back({p + "linear_attn.in_proj_qkv.weight",
                       {(int64_t)cfg.lin_qkv_dim(), (int64_t)cfg.hidden}, 1});
      needs.push_back({p + "linear_attn.in_proj_z.weight",
                       {(int64_t)cfg.lin_value_dim(), (int64_t)cfg.hidden},
                       1});
      needs.push_back({p + "linear_attn.in_proj_a.weight",
                       {(int64_t)cfg.lin_v_heads, (int64_t)cfg.hidden}, 1});
      needs.push_back({p + "linear_attn.in_proj_b.weight",
                       {(int64_t)cfg.lin_v_heads, (int64_t)cfg.hidden}, 1});
      needs.push_back({p + "linear_attn.conv1d.weight", {}, 2});
      needs.push_back({p + "linear_attn.out_proj.weight",
                       {(int64_t)cfg.hidden, (int64_t)cfg.lin_value_dim()},
                       1});
      needs.push_back({p + "linear_attn.norm.weight",
                       {(int64_t)cfg.lin_v_dim}, 0});
      // NOTE: the MOSS-Image-SFT checkpoint stores these two vectors WITHOUT
      // the .weight suffix (verified against the published safetensors
      // index: BF16 [32] each).
      needs.push_back({p + "linear_attn.A_log",
                       {(int64_t)cfg.lin_v_heads}, 0});
      needs.push_back({p + "linear_attn.dt_bias",
                       {(int64_t)cfg.lin_v_heads}, 0});
    }
  }

  // Header maps for all shards (small: JSON only, no payloads).
  struct Shard {
    std::string path;
    std::map<std::string, Impl::SafeTensorInfo> map;
  };
  std::vector<Shard> shards;
  for (const auto& f : files) {
    uint64_t hs = 0;
    shards.push_back({f, Impl::read_safetensors_header(f, hs)});
  }
  auto find = [&shards](const std::string& n, const Shard*& s,
                        const Impl::SafeTensorInfo*& info) -> bool {
    for (const auto& sh : shards) {
      auto it = sh.map.find(n);
      if (it != sh.map.end()) {
        s = &sh;
        info = &it->second;
        return true;
      }
    }
    return false;
  };

  // Group needs by shard so each file is opened once (streaming reads).
  std::map<size_t, std::vector<const Need*>> by_shard;
  for (const auto& nd : needs) {
    const Shard* s = nullptr;
    const Impl::SafeTensorInfo* info = nullptr;
    if (!find(nd.name, s, info))
      throw std::runtime_error("tensor missing from shards: " + nd.name);
    by_shard[s - shards.data()].push_back(&nd);
  }

  // Resolve each Need to its destination view (rank-preserving).
  auto dest1 = [&w, &cfg](const Need& nd) -> typename Weights<ExecSpace,
                                                              Scalar>::View1 {
    using V1 = typename Weights<ExecSpace, Scalar>::View1;
    if (nd.name == "language_model.norm.weight") return w.final_norm;
    // layer norms / linear vectors: parse layer index from name.
    const std::string pre = "language_model.layers.";
    size_t b = pre.size(), e = nd.name.find('.', b);
    int l = std::stoi(nd.name.substr(b, e - b));
    const std::string tail = nd.name.substr(e + 1);
    typename Weights<ExecSpace, Scalar>::Layer& L = w.layers[(size_t)l];
    if (tail == "input_layernorm.weight") return L.input_norm;
    if (tail == "post_attention_layernorm.weight") return L.post_norm;
    if (tail == "self_attn.q_norm.weight") return L.full.q_norm;
    if (tail == "self_attn.k_norm.weight") return L.full.k_norm;
    if (tail == "linear_attn.norm.weight") return L.lin.norm;
    if (tail == "linear_attn.A_log") return L.lin.a_log;
    if (tail == "linear_attn.dt_bias") return L.lin.dt_bias;
    throw std::runtime_error("no destination: " + nd.name);
    return V1();
  };
  auto dest2 = [&w, &cfg](const Need& nd) -> typename Weights<ExecSpace,
                                                              Scalar>::View2 {
    using V2 = typename Weights<ExecSpace, Scalar>::View2;
    if (nd.name == "language_model.embed_tokens.weight") return w.embed;
    const std::string pre = "language_model.layers.";
    size_t b = pre.size(), e = nd.name.find('.', b);
    int l = std::stoi(nd.name.substr(b, e - b));
    const std::string tail = nd.name.substr(e + 1);
    typename Weights<ExecSpace, Scalar>::Layer& L = w.layers[(size_t)l];
    if (tail == "mlp.gate_proj.weight") return L.gate_proj;
    if (tail == "mlp.up_proj.weight") return L.up_proj;
    if (tail == "mlp.down_proj.weight") return L.down_proj;
    if (tail == "self_attn.q_proj.weight") return L.full.q_proj;
    if (tail == "self_attn.k_proj.weight") return L.full.k_proj;
    if (tail == "self_attn.v_proj.weight") return L.full.v_proj;
    if (tail == "self_attn.o_proj.weight") return L.full.o_proj;
    if (tail == "linear_attn.in_proj_qkv.weight") return L.lin.in_qkv;
    if (tail == "linear_attn.in_proj_z.weight") return L.lin.in_z;
    if (tail == "linear_attn.in_proj_a.weight") return L.lin.in_a;
    if (tail == "linear_attn.in_proj_b.weight") return L.lin.in_b;
    if (tail == "linear_attn.conv1d.weight" || tail == "conv1d")
      return L.lin.conv;
    if (tail == "linear_attn.out_proj.weight") return L.lin.out_proj;
    throw std::runtime_error("no destination: " + nd.name);
    return V2();
  };

  std::vector<char> raw;
  std::vector<float> host;
  for (const auto& kv : by_shard) {
    const Shard& sh = shards[kv.first];
    std::ifstream f(sh.path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + sh.path);
    for (const Need* ndp : kv.second) {
      const Need& nd = *ndp;
      auto it = sh.map.find(nd.name);
      const Impl::SafeTensorInfo& info = it->second;
      if (nd.kind == 2) {
        // Conv1d weight [C,1,K] -> stored [C,K] (middle extent must be 1).
        if (info.shape.size() != 3 || info.shape[1] != 1 ||
            info.shape[0] != cfg.lin_qkv_dim() || info.shape[2] != cfg.conv_k)
          throw std::runtime_error("tensor " + nd.name +
                                   ": bad conv shape");
        int64_t numel = info.shape[0] * info.shape[2];
        raw.resize((size_t)info.nbytes);
        f.seekg((std::streamoff)info.data_begin);
        f.read(raw.data(), (std::streamsize)info.nbytes);
        if (!f) throw std::runtime_error("read failed: " + nd.name);
        Impl::convert_payload_to_float(nd.name, info, raw.data(), numel,
                                       host);
        Impl::fill_from_float<ExecSpace, Scalar>(dest2(nd), host, nd.name);
        continue;
      }
      Impl::require_shape(nd.name, info, nd.shape);
      int64_t numel = 1;
      for (int64_t d : nd.shape) numel *= d;
      raw.resize((size_t)info.nbytes);
      f.seekg((std::streamoff)info.data_begin);
      f.read(raw.data(), (std::streamsize)info.nbytes);
      if (!f) throw std::runtime_error("read failed: " + nd.name);
      Impl::convert_payload_to_float(nd.name, info, raw.data(), numel, host);
      if (nd.kind == 0)
        Impl::fill_from_float<ExecSpace, Scalar>(dest1(nd), host, nd.name);
      else
        Impl::fill_from_float<ExecSpace, Scalar>(dest2(nd), host, nd.name);
    }
  }
}
// ---- encoder forward --------------------------------------------------------
// One full-attention decoder layer on h [T, H] (in-place residual updates).
// Reference: Qwen3_5DecoderLayer (full path) + Qwen3_5Attention.
template <class ExecSpace, class Scalar, class HView>
void forward_full_layer(
    const DNNHandle<ExecSpace>& handle, const Config& cfg,
    const typename Weights<ExecSpace, Scalar>::Layer& L,
    const Kokkos::View<float*, ExecSpace>& rope_inv, HView& h) {
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using UView2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                              Kokkos::MemoryUnmanaged>;
  using UView3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                              Kokkos::MemoryUnmanaged>;
  const size_t T = h.extent(0), H = (size_t)cfg.hidden;
  const size_t Nq = (size_t)cfg.num_q_heads, Nk = (size_t)cfg.num_kv_heads;
  const size_t Hd = (size_t)cfg.head_dim, Qd = Nq * Hd;
  const size_t I = (size_t)cfg.intermediate;
  View2 n("Qwen35::n", T, H);
  rmsnorm_one_plus<ExecSpace, Scalar>(h, L.input_norm, cfg.rms_eps, n,
                                     "Qwen35::in_norm");
  View2 qg("Qwen35::qg", T, 2 * Qd);
  gemm(handle, Scalar(1), n, L.full.q_proj, Scalar(0), qg, "Qwen35::q");
  View2 k2("Qwen35::k", T, Nk * Hd), v2("Qwen35::v", T, Nk * Hd);
  gemm(handle, Scalar(1), n, L.full.k_proj, Scalar(0), k2, "Qwen35::k");
  gemm(handle, Scalar(1), n, L.full.v_proj, Scalar(0), v2, "Qwen35::v");
  // Pack the query/gate slices into contiguous buffers (same stride
  // rationale as the q/k/v packs below): qg rows are 2*Qd wide.
  //
  // The reference splits the query projection PER HEAD, not as a global
  // [q | gate] halves: q_proj(h) is viewed as [T, Nq, 2*Hd] and chunked on
  // the last axis (modeling_qwen3_5.py Qwen3_5Attention.forward:
  //   query_states, gate = torch.chunk(q_proj(h).view(*shape, -1, 2*Hd),
  //                                    2, dim=-1))
  // so head h's query occupies qg[t, h*2*Hd : h*2*Hd + Hd] and its gate
  // occupies the following Hd columns. A global [0, Qd)/[Qd, 2*Qd) split
  // silently mixes head boundaries and diverges from torch.
  View2 q2("Qwen35::q", T, Qd), gate2("Qwen35::gate", T, Qd);
  Kokkos::parallel_for(
      "Qwen35::pack_qg",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Qd}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        const size_t h = d / Hd, dd = d % Hd;
        q2(t, d) = qg(t, h * 2 * Hd + dd);
        gate2(t, d) = qg(t, h * 2 * Hd + Hd + dd);
      });
  Kokkos::fence();
  UView3 q(q2.data(), T, Nq, Hd);
  UView2 gate(gate2.data(), T, Qd);
  Impl::qk_rmsnorm_oneplus_3d<ExecSpace, Scalar>(q, L.full.q_norm, cfg.rms_eps,
                                                 q, "Qwen35::qn");
  UView3 k(k2.data(), T, Nk, Hd);
  UView3 v(v2.data(), T, Nk, Hd);
  Impl::qk_rmsnorm_oneplus_3d<ExecSpace, Scalar>(k, L.full.k_norm, cfg.rms_eps,
                                                k, "Qwen35::kn");
  Impl::qwen_partial_rope_inplace<ExecSpace, Scalar>(q, rope_inv,
                                                    cfg.rotary_dim(),
                                                    "Qwen35::rope_q");
  Impl::qwen_partial_rope_inplace<ExecSpace, Scalar>(k, rope_inv,
                                                    cfg.rotary_dim(),
                                                    "Qwen35::rope_k");
  View2 ao("Qwen35::ao", T, Nq * Hd);
  UView3 ao3(ao.data(), T, Nq, Hd);
  const float scale =
      1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
  attention_gqa_single<ExecSpace, Scalar>(handle, q, k, v, 0, ao3, scale,
                                          "Qwen35::attn");
  Impl::qwen_gate_multiply<ExecSpace, Scalar>(ao, gate, ao, "Qwen35::gate");
  View2 o("Qwen35::o", T, H);
  gemm(handle, Scalar(1), ao, L.full.o_proj, Scalar(0), o, "Qwen35::o");
  Impl::qwen_residual_add<ExecSpace, Scalar>(h, o, "Qwen35::res1");
  // MLP: down(silu(gate) * up).
  rmsnorm_one_plus<ExecSpace, Scalar>(h, L.post_norm, cfg.rms_eps, n,
                                     "Qwen35::post_norm");
  View2 g("Qwen35::g", T, I), u("Qwen35::u", T, I);
  gemm(handle, Scalar(1), n, L.gate_proj, Scalar(0), g, "Qwen35::mg");
  gemm(handle, Scalar(1), n, L.up_proj, Scalar(0), u, "Qwen35::mu");
  swiglu<ExecSpace, Scalar>(g, u, g, "Qwen35::swiglu");
  View2 d("Qwen35::d", T, H);
  gemm(handle, Scalar(1), g, L.down_proj, Scalar(0), d, "Qwen35::md");
  Impl::qwen_residual_add<ExecSpace, Scalar>(h, d, "Qwen35::res2");
}

// One linear-attention (Gated DeltaNet) decoder layer on h [T, H].
// Reference: Qwen3_5DecoderLayer (linear path) + Qwen3_5GatedDeltaNet
// (torch fallbacks: grouped causal conv + SiLU, gates, recurrent rule,
// gated norm).
template <class ExecSpace, class Scalar, class HView>
void forward_linear_layer(
    const DNNHandle<ExecSpace>& handle, const Config& cfg,
    const typename Weights<ExecSpace, Scalar>::Layer& L, HView& h) {
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using UView2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                              Kokkos::MemoryUnmanaged>;
  using UView3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                              Kokkos::MemoryUnmanaged>;
  using FView2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using FView3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
  const size_t T = h.extent(0), H = (size_t)cfg.hidden;
  const size_t Kd = (size_t)cfg.lin_key_dim(), Vd = (size_t)cfg.lin_value_dim();
  const size_t Kh = (size_t)cfg.lin_k_heads, Vh = (size_t)cfg.lin_v_heads;
  const size_t Kdh = (size_t)cfg.lin_k_dim, Vdh = (size_t)cfg.lin_v_dim;
  const size_t I = (size_t)cfg.intermediate;
  const int rep = cfg.lin_v_heads / cfg.lin_k_heads;
  View2 n("Qwen35::n", T, H);
  rmsnorm_one_plus<ExecSpace, Scalar>(h, L.input_norm, cfg.rms_eps, n,
                                     "Qwen35::in_norm");
  View2 qkv("Qwen35::qkv", T, Kd * 2 + Vd);
  gemm(handle, Scalar(1), n, L.lin.in_qkv, Scalar(0), qkv, "Qwen35::qkv");
  View2 z("Qwen35::z", T, Vd), a("Qwen35::a", T, Vh), b("Qwen35::b", T, Vh);
  gemm(handle, Scalar(1), n, L.lin.in_z, Scalar(0), z, "Qwen35::z");
  gemm(handle, Scalar(1), n, L.lin.in_a, Scalar(0), a, "Qwen35::a");
  gemm(handle, Scalar(1), n, L.lin.in_b, Scalar(0), b, "Qwen35::b");
  // Depthwise causal conv k + SiLU (reference applies silu after conv).
  View2 cv("Qwen35::cv", T, Kd * 2 + Vd);
  View1 no_bias;
  depthwise_conv1d_causal_forward<ExecSpace, Scalar>(qkv, L.lin.conv, no_bias,
                                                    cv, "Qwen35::conv");
  silu_forward<ExecSpace, Scalar>(cv, cv, "Qwen35::conv_silu");
  // Pack the q/k/v slices into contiguous buffers: cv rows are
  // (2*Kd+Vd) wide, so rank-3 views constructed directly over cv would
  // stride every row past the first incorrectly (same as q/gate above).
  // Explicit elementwise copies (subview pair ranges proved unreliable).
  View2 q16p("Qwen35::q16", T, Kd), k16p("Qwen35::k16", T, Kd),
      vvp("Qwen35::vv", T, Vd);
  Kokkos::parallel_for(
      "Qwen35::pack_qkv",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Kd}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        q16p(t, d) = cv(t, d);
        k16p(t, d) = cv(t, Kd + d);
      });
  Kokkos::parallel_for(
      "Qwen35::pack_v",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Vd}),
      KOKKOS_LAMBDA(size_t t, size_t d) { vvp(t, d) = cv(t, 2 * Kd + d); });
  Kokkos::fence();
  UView3 q16(q16p.data(), T, Kh, Kdh);
  UView3 k16(k16p.data(), T, Kh, Kdh);
  UView3 vv(vvp.data(), T, Vh, Vdh);
  UView3 z3(z.data(), T, Vh, Vdh);
  FView2 beta("Qwen35::beta", T, Vh), gg("Qwen35::g", T, Vh);
  Impl::qwen_gdn_gates<ExecSpace, Scalar>(a, b, L.lin.a_log, L.lin.dt_bias,
                                         beta, gg, "Qwen35::gates");
  View2 qe("Qwen35::qe", T, Vh * Kdh), ke("Qwen35::ke", T, Vh * Kdh);
  UView3 q32(qe.data(), T, Vh, Kdh);
  UView3 k32(ke.data(), T, Vh, Kdh);
  Impl::qwen_head_repeat<ExecSpace, Scalar>(q16, q32, rep, "Qwen35::qrep");
  Impl::qwen_head_repeat<ExecSpace, Scalar>(k16, k32, rep, "Qwen35::krep");
  Impl::qwen_qk_l2norm<ExecSpace, Scalar>(
      q32, 1e-6f, 1.0f / std::sqrt(static_cast<float>(cfg.lin_k_dim)),
      "Qwen35::qnorm");
  Impl::qwen_qk_l2norm<ExecSpace, Scalar>(k32, 1e-6f, 1.0f, "Qwen35::knorm");
  FView3 state("Qwen35::state", Vh, Kdh, Vdh);
  Kokkos::deep_copy(state, 0.0f);
  View2 coref("Qwen35::core", T, Vd);
  UView3 core(coref.data(), T, Vh, Vdh);
  Impl::qwen_gated_delta_recurrent<ExecSpace, Scalar>(q32, k32, vv, beta, gg,
                                                     state, core,
                                                     "Qwen35::gdn");
  Impl::qwen_rmsnorm_gated<ExecSpace, Scalar>(core, z3, L.lin.norm, cfg.rms_eps,
                                             core, "Qwen35::gnorm");
  View2 o("Qwen35::o", T, H);
  gemm(handle, Scalar(1), coref, L.lin.out_proj, Scalar(0), o, "Qwen35::o");
  Impl::qwen_residual_add<ExecSpace, Scalar>(h, o, "Qwen35::res1");
  // MLP (shared with full layers).
  rmsnorm_one_plus<ExecSpace, Scalar>(h, L.post_norm, cfg.rms_eps, n,
                                     "Qwen35::post_norm");
  View2 g("Qwen35::g", T, I), u("Qwen35::u", T, I);
  gemm(handle, Scalar(1), n, L.gate_proj, Scalar(0), g, "Qwen35::mg");
  gemm(handle, Scalar(1), n, L.up_proj, Scalar(0), u, "Qwen35::mu");
  swiglu<ExecSpace, Scalar>(g, u, g, "Qwen35::swiglu");
  View2 d("Qwen35::d", T, H);
  gemm(handle, Scalar(1), g, L.down_proj, Scalar(0), d, "Qwen35::md");
  Impl::qwen_residual_add<ExecSpace, Scalar>(h, d, "Qwen35::res2");
}
// Text-only encoder model: embed -> 32x decoder layer -> final norm.
// Consumes token ids [B, S] (NOT raw text); per-sample unpadded handling
// with explicit host-side valid lengths (matches pipe.encode_prompt, which
// runs one unpadded forward per prompt and zero-pads afterwards).
template <class ExecSpace, class Scalar>
struct EncoderModel {
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using UView2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                              Kokkos::MemoryUnmanaged>;
  using IdView = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> w;
  Kokkos::View<float*, ExecSpace> rope_inv;  // [rotary_dim/2]

  explicit EncoderModel(const Config& c = Config::moss_image_sft())
      : cfg(c), w(c) {
    auto hv = Impl::qwen_rope_inv_host(cfg.rotary_dim(), cfg.rope_theta);
    rope_inv = Kokkos::View<float*, ExecSpace>("Qwen35::rope_inv", hv.size());
    auto hm = Kokkos::create_mirror_view(rope_inv);
    for (size_t i = 0; i < hv.size(); ++i) hm(i) = hv[i];
    Kokkos::deep_copy(rope_inv, hm);
  }

  void init_random(unsigned seed = 1234) { w.random_init(seed); }
  void load_safetensors(const std::vector<std::string>& files) {
    load_weights_safetensors(w, files);
  }
  void release() {
    w.release();
    rope_inv = Kokkos::View<float*, ExecSpace>();
  }
  size_t resident_bytes() const {
    return w.resident_bytes() + rope_inv.extent(0) * sizeof(float);
  }

  // Forward: ids [B, S]; lengths[b] valid tokens in row b (1..S);
  // out [B, S, H] receives hidden states on valid rows (padding rows are
  // left untouched, so callers may pre-zero to reproduce the pipeline's
  // zero-padded batch). Ids are range-checked on the host ([0, V)).
  void forward(const DNNHandle<ExecSpace>& handle, const IdView& ids,
               const std::vector<size_t>& lengths, const View3& out) {
    const size_t B = ids.extent(0), S = ids.extent(1);
    if (out.extent(0) != B || out.extent(1) != S ||
        out.extent(2) != (size_t)cfg.hidden)
      throw std::invalid_argument("EncoderModel::forward: dim mismatch");
    if (lengths.size() != B)
      throw std::invalid_argument("EncoderModel::forward: lengths size");
    if (w.layers.empty())
      throw std::logic_error("EncoderModel::forward: weights empty");
    // Host-side id range check (one mirror per forward; negligible vs a
    // 9B encoder pass, and required for gather safety).
    {
      auto hm = Kokkos::create_mirror_view(ids);
      Kokkos::deep_copy(hm, ids);
      for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
          if (hm(b, s) < 0 || hm(b, s) >= cfg.vocab_size)
            throw std::invalid_argument("EncoderModel::forward: id range");
    }
    Kokkos::View<int*, ExecSpace> toks("Qwen35::toks", S);
    for (size_t b = 0; b < B; ++b) {
      const size_t T = lengths[b];
      if (T == 0 || T > S)
        throw std::invalid_argument("EncoderModel::forward: bad length");
      Kokkos::View<int*, ExecSpace, Kokkos::MemoryUnmanaged> idpre(
          ids.data() + b * S, T);
      Kokkos::deep_copy(
          Kokkos::subview(toks, std::make_pair(size_t{0}, T)), idpre);
      // Note: subview(ids, b, ALL) on LayoutRight [B, S] is strided, so
      // the valid prefix is repacked above via the contiguous unmanaged
      // row-prefix view (not via a strided subview copy).
      auto tpre = Kokkos::subview(toks, std::make_pair(size_t{0}, T));
      View2 h("Qwen35::h", T, cfg.hidden);
      embedding_gather<ExecSpace, Scalar>(w.embed, tpre, h, "Qwen35::embed");
      for (int l = 0; l < cfg.num_layers; ++l) {
        if (cfg.is_full_layer(l))
          forward_full_layer<ExecSpace, Scalar>(handle, cfg, w.layers[(size_t)l],
                                                rope_inv, h);
        else
          forward_linear_layer<ExecSpace, Scalar>(handle, cfg,
                                                  w.layers[(size_t)l], h);
      }
      View2 hf("Qwen35::hf", T, cfg.hidden);
      // Final norm is Qwen3_5RMSNorm, i.e. the (1+w) variant (zero-init
      // weight), same formula as the input/post norms.
      rmsnorm_one_plus<ExecSpace, Scalar>(h, w.final_norm, cfg.rms_eps, hf,
                                         "Qwen35::final_norm");
      UView2 dst(out.data() + b * S * (size_t)cfg.hidden, T,
                 (size_t)cfg.hidden);
      Kokkos::deep_copy(dst, hf);
    }
    Kokkos::fence();
  }

  // All-valid convenience overload (lengths[b] = S).
  void forward(const DNNHandle<ExecSpace>& handle, const IdView& ids,
               const View3& out) {
    forward(handle, ids, std::vector<size_t>(ids.extent(0), ids.extent(1)),
            out);
  }
};

}  // namespace Qwen35
}  // namespace KokkosDNN

#endif
