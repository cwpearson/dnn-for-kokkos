#ifndef KOKKOSDNN_SANACONV_HPP
#define KOKKOSDNN_SANACONV_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_PatchEmbed.hpp"
#include "KokkosDNN_SwiGLU.hpp"

// Sana GLUMBConv MixFFN ops (NVlabs Sana, diffusion/model/nets/basic_modules.py
// classes ConvLayer / GLUMBConv), plus the Sana patch_size=1 patch embed and
// its inverse.
//
// Reference pipeline (per token row t = h*W + w, single batch element):
//   u  = SiLU(x @ inv_w^T + inv_b)          // inverted 1x1 conv, in -> 2*hid
//   dw = depthwise3x3(u)                    // groups=2*hid, pad=1, stride=1
//   (a, g) = chunk(dw, 2); h = a * SiLU(g)  // GLU-style gate
//   y  = h @ pt_w^T                         // point 1x1 conv, hid -> out
// with mlp_acts (silu, silu, None): SiLU after the inverted conv AND SiLU on
// the gate, no activation after the point conv. Inverted conv has bias,
// depthwise conv has bias, point conv has no bias and no norm.
// Sana-600M proportions: in=1152, hidden=2880 (mlp_ratio 2.5), so the
// inverted conv emits 2*hidden=5760 channels and the depthwise conv has
// groups=5760.
//
// Weight layout (documented choice, verified against KokkosDNN_GemmBias.hpp
// and KokkosDNN_PatchEmbed.hpp): 1x1 conv weights are [Cout, Cin] row-major,
// i.e. the HF Linear [out_features, in_features] convention with NO explicit
// transpose (gemm computes Y = X * W^T). This matches gemm/gemm_bias
// (W:[N,K]), dense_forward (W:[Out,In]), patchembed_forward (weight
// [D,C*K*K], one row per output channel), and the Llama wq-style call sites.
// Depthwise weights are [Cin, 3, 3] LayoutRight (one 3x3 kernel per channel).
// Token views are LayoutRight [T, C] with T = H*W row-major (t = h*W + w),
// or batched [B, T, C]. All accumulation is FP32, cast back to Scalar.
// Backend-portable: only parallel_for MDRange/Range kernels plus gemm_bias
// (which owns the cuBLAS/TPL dispatch ladder); no backend-specific code.

namespace KokkosDNN {
namespace Impl {

// Elementwise SiLU in place over a rank-2 view (FP32, cast back).
// Reuses KokkosDNN::silu from KokkosDNN_SwiGLU.hpp (same x*sigmoid(x)).
template <class ExecSpace, class Scalar>
void silu_inplace_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& v,
    const std::string& label) {
  const size_t R = v.extent(0), C = v.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        v(i, j) = static_cast<Scalar>(silu(static_cast<float>(v(i, j))));
      });
}

// GLU-style gated multiply: h(t,j) = dw(t,j) * SiLU(dw(t,j+Hid)).
// Each output depends only on same-row inputs; dw and h must not alias
// (h is narrower than dw, so out-of-place is required).
template <class ExecSpace, class Scalar>
void silu_gate_multiply_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dw,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& h, size_t Hid,
    const std::string& label) {
  const size_t T = h.extent(0);
  if (T == 0 || Hid == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {T, Hid}),
      KOKKOS_LAMBDA(size_t t, size_t j) {
        const float a = static_cast<float>(dw(t, j));
        const float g = static_cast<float>(dw(t, j + Hid));
        h(t, j)       = static_cast<Scalar>(a * silu(g));
      });
}

// Depthwise 3x3, stride 1, pad 1 (zero padding, i.e. PyTorch Conv2d with
// padding=1 / Sana get_same_padding(3) = 1), groups=Cin. Rank-2 token path:
// x/y [T, C] with T = H*W row-major, w [C, 3, 3], b [C] (or empty).
template <class ExecSpace, class Scalar>
void depthwise3x3_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t T = x.extent(0), C = x.extent(1);
  if (T == 0 || C == 0) return;
  const bool biased = (b.extent(0) != 0);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = static_cast<int>(h) + kh - 1;
          if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = static_cast<int>(ww) + kw - 1;
            if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
            const size_t s =
                static_cast<size_t>(ih) * W + static_cast<size_t>(iw);
            acc += static_cast<float>(x(s, c)) *
                   static_cast<float>(w(c, static_cast<size_t>(kh),
                                          static_cast<size_t>(kw)));
          }
        }
        y(t, c) = static_cast<Scalar>(acc);
      });
}

// Batched rank-3 variant: x/y [B, T, C], weights shared across the batch.
template <class ExecSpace, class Scalar>
void depthwise3x3_3d(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t B = x.extent(0), T = x.extent(1), C = x.extent(2);
  if (B == 0 || T == 0 || C == 0) return;
  const bool biased = (b.extent(0) != 0);
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, C}),
      KOKKOS_LAMBDA(size_t bb, size_t t, size_t c) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = static_cast<int>(h) + kh - 1;
          if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = static_cast<int>(ww) + kw - 1;
            if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
            const size_t s =
                static_cast<size_t>(ih) * W + static_cast<size_t>(iw);
            acc += static_cast<float>(x(bb, s, c)) *
                   static_cast<float>(w(c, static_cast<size_t>(kh),
                                          static_cast<size_t>(kw)));
          }
        }
        y(bb, t, c) = static_cast<Scalar>(acc);
      });
}

}  // namespace Impl

// 1x1 convolution as biased GEMM: y = x @ w^T + b with x [T, Cin],
// w [Cout, Cin] (HF Linear layout, no transpose), b [Cout] broadcast over
// rows (or empty extent-0 view for no bias, e.g. the GLUMBConv point conv),
// y [T, Cout]. Thin wrapper over gemm_bias so the cuBLAS/TPL dispatch
// ladder applies unchanged; FP32 accumulation, cast back to Scalar.
template <class ExecSpace, class Scalar>
void conv1x1_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::conv1x1_forward") {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (w.extent(1) != Cin || y.extent(0) != T || y.extent(1) != Cout ||
      (biased && b.extent(0) != Cout))
    throw std::invalid_argument("conv1x1_forward: dimension mismatch");
  if (T == 0 || Cin == 0 || Cout == 0) return;
  gemm_bias(handle, x, w, b, y, label);
}

// Depthwise 3x3 convolution, stride 1, pad 1 (zero padding), groups=Cin:
//   y(t,c) = b(c) + sum_{kh,kw} w(c,kh,kw) * x_pad(h+kh-1, w+kw-1, c)
// with t = h*W + w. x/y [T, C] LayoutRight with T = H*W, w [C, 3, 3]
// LayoutRight, b [C] or empty (extent 0) for no bias. Scalar-agnostic;
// FP32 accumulation. Backend-portable MDRange kernel.
template <class ExecSpace, class Scalar>
void depthwise_conv3x3_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::depthwise_conv3x3_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("depthwise_conv3x3_forward: H and W must be nonzero");
  const size_t T = x.extent(0), C = x.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (T != H * W || w.extent(0) != C || w.extent(1) != 3 || w.extent(2) != 3 ||
      y.extent(0) != T || y.extent(1) != C || (biased && b.extent(0) != C))
    throw std::invalid_argument("depthwise_conv3x3_forward: dimension mismatch");
  if (T == 0 || C == 0) return;
  Impl::depthwise3x3_2d<ExecSpace, Scalar>(x, w, b, y, H, W, label);
}

// Batched variant: x/y [B, T, C], weights shared across the batch.
template <class ExecSpace, class Scalar>
void depthwise_conv3x3_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::depthwise_conv3x3_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("depthwise_conv3x3_forward: H and W must be nonzero");
  const size_t B = x.extent(0), T = x.extent(1), C = x.extent(2);
  const bool biased = (b.extent(0) != 0);
  if (T != H * W || w.extent(0) != C || w.extent(1) != 3 || w.extent(2) != 3 ||
      y.extent(0) != B || y.extent(1) != T || y.extent(2) != C ||
      (biased && b.extent(0) != C))
    throw std::invalid_argument("depthwise_conv3x3_forward: dimension mismatch");
  if (B == 0 || T == 0 || C == 0) return;
  Impl::depthwise3x3_3d<ExecSpace, Scalar>(x, w, b, y, H, W, label);
}

// GLUMBConv MixFFN forward (rank-2, single batch element or flattened
// batch): x [T, Cin] with T = H*W, y [T, Cout], where
//   inv_w [2*Hid, Cin], inv_b [2*Hid] (or empty for no bias),
//   dw_w [2*Hid, 3, 3], dw_b [2*Hid] (or empty for no bias),
//   pt_w [Cout, Hid] (point conv is unbiased).
// Pipeline: u = SiLU(x @ inv_w^T + inv_b); dw = depthwise3x3(u);
// split dw into (a, g) along channels; h = a * SiLU(g);
// y = h @ pt_w^T. Scratch: [T, 2*Hid] x2 + [T, Hid].
template <class ExecSpace, class Scalar>
void glumbconv_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& inv_w,
    const Kokkos::View<Scalar*, ExecSpace>& inv_b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& dw_w,
    const Kokkos::View<Scalar*, ExecSpace>& dw_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& pt_w,
    size_t H, size_t W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::glumbconv_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("glumbconv_forward: H and W must be nonzero");
  const size_t T = x.extent(0), Cin = x.extent(1);
  const size_t inv_out = inv_w.extent(0);
  const bool inv_biased = (inv_b.extent(0) != 0);
  const bool dw_biased  = (dw_b.extent(0) != 0);
  if (T != H * W || inv_out == 0 || inv_out % 2 != 0 || inv_w.extent(1) != Cin ||
      (inv_biased && inv_b.extent(0) != inv_out) || dw_w.extent(0) != inv_out ||
      dw_w.extent(1) != 3 || dw_w.extent(2) != 3 ||
      (dw_biased && dw_b.extent(0) != inv_out))
    throw std::invalid_argument("glumbconv_forward: dimension mismatch");
  const size_t Hid = inv_out / 2, Cout = pt_w.extent(0);
  if (pt_w.extent(1) != Hid || y.extent(0) != T || y.extent(1) != Cout)
    throw std::invalid_argument("glumbconv_forward: dimension mismatch");
  if (T == 0 || Cin == 0 || Cout == 0) return;

  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> u(label + "_u", T,
                                                           inv_out);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dw(label + "_dw", T,
                                                            inv_out);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> h(label + "_h", T,
                                                           Hid);
  gemm_bias(handle, x, inv_w, inv_b, u, label + "_inv");
  Impl::silu_inplace_2d<ExecSpace, Scalar>(u, label + "_silu");
  Impl::depthwise3x3_2d<ExecSpace, Scalar>(u, dw_w, dw_b, dw, H, W,
                                          label + "_dw");
  Impl::silu_gate_multiply_2d<ExecSpace, Scalar>(dw, h, Hid, label + "_gate");
  Kokkos::View<Scalar*, ExecSpace> no_bias;  // point conv is unbiased
  gemm_bias(handle, h, pt_w, no_bias, y, label + "_pt");
}

// Batched variant: x [B, T, Cin], y [B, T, Cout], weights shared across the
// batch. GEMM stages run on flattened [B*T, C] aliases of contiguous views
// (same aliasing pattern as patchembed_forward); the depthwise stage has a
// dedicated rank-3 kernel. x and y must be contiguous.
template <class ExecSpace, class Scalar>
void glumbconv_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& inv_w,
    const Kokkos::View<Scalar*, ExecSpace>& inv_b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& dw_w,
    const Kokkos::View<Scalar*, ExecSpace>& dw_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& pt_w,
    size_t H, size_t W,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::glumbconv_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("glumbconv_forward: H and W must be nonzero");
  const size_t B = x.extent(0), T = x.extent(1), Cin = x.extent(2);
  const size_t inv_out = inv_w.extent(0);
  const bool inv_biased = (inv_b.extent(0) != 0);
  const bool dw_biased  = (dw_b.extent(0) != 0);
  if (T != H * W || inv_out == 0 || inv_out % 2 != 0 || inv_w.extent(1) != Cin ||
      (inv_biased && inv_b.extent(0) != inv_out) || dw_w.extent(0) != inv_out ||
      dw_w.extent(1) != 3 || dw_w.extent(2) != 3 ||
      (dw_biased && dw_b.extent(0) != inv_out))
    throw std::invalid_argument("glumbconv_forward: dimension mismatch");
  const size_t Hid = inv_out / 2, Cout = pt_w.extent(0);
  if (pt_w.extent(1) != Hid || y.extent(0) != B || y.extent(1) != T ||
      y.extent(2) != Cout)
    throw std::invalid_argument("glumbconv_forward: dimension mismatch");
  if (!x.span_is_contiguous() || !y.span_is_contiguous())
    throw std::invalid_argument("glumbconv_forward: x and y must be contiguous");
  if (B == 0 || T == 0 || Cin == 0 || Cout == 0) return;

  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> u(label + "_u", B, T,
                                                            inv_out);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> dw(label + "_dw", B,
                                                             T, inv_out);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> h(label + "_h", B, T,
                                                            Hid);
  // Flattened aliases (fresh views above are contiguous by construction).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> xf(x.data(), B * T,
                                                            Cin);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> uf(u.data(), B * T,
                                                            inv_out);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dwf(dw.data(), B * T,
                                                             inv_out);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> hf(h.data(), B * T,
                                                            Hid);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> yf(y.data(), B * T,
                                                            Cout);
  gemm_bias(handle, xf, inv_w, inv_b, uf, label + "_inv");
  Impl::silu_inplace_2d<ExecSpace, Scalar>(uf, label + "_silu");
  Impl::depthwise3x3_3d<ExecSpace, Scalar>(u, dw_w, dw_b, dw, H, W,
                                          label + "_dw");
  Impl::silu_gate_multiply_2d<ExecSpace, Scalar>(dwf, hf, Hid, label + "_gate");
  Kokkos::View<Scalar*, ExecSpace> no_bias;  // point conv is unbiased
  gemm_bias(handle, hf, pt_w, no_bias, yf, label + "_pt");
}

// Sana patch embed (patch_size=1, stride 1): 1x1 conv over VAE latents
// images [B, C, H, W] -> tokens [B, T, D] with T = H*W, weight [D, C],
// bias [D] or empty. Delegates to patchembed_forward with K=1, S=1 (whose
// im2col degenerates to a transpose and whose weight layout [D, C*1*1] is
// exactly [D, C]). Sana-600M case: latents [B, 32, 32, 32] -> [B, 1024, 1152]
// with weight [1152, 32].
template <class ExecSpace, class Scalar>
void patch_embed_1x1_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& tokens,
    const std::string& label = "KokkosDNN::patch_embed_1x1_forward") {
  patchembed_forward(handle, images, weight, bias, tokens, 1, 1, label);
}

// Unpatchify (inverse of the patch embed): tokens [T, C] -> image [C, H, W]
// with T = H*W row-major (token t = h*W + w). Pure transpose, no handle.
template <class ExecSpace, class Scalar>
void unpatchify_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& tokens,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& image,
    size_t H, size_t W,
    const std::string& label = "KokkosDNN::unpatchify_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("unpatchify_forward: H and W must be nonzero");
  const size_t T = tokens.extent(0), C = tokens.extent(1);
  if (T != H * W || image.extent(0) != C || image.extent(1) != H ||
      image.extent(2) != W)
    throw std::invalid_argument("unpatchify_forward: dimension mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        image(c, t / W, t % W) = tokens(t, c);
      });
}

// Batched variant: tokens [B, T, C] -> images [B, C, H, W].
template <class ExecSpace, class Scalar>
void unpatchify_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& tokens,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images,
    size_t H, size_t W,
    const std::string& label = "KokkosDNN::unpatchify_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("unpatchify_forward: H and W must be nonzero");
  const size_t B = tokens.extent(0), T = tokens.extent(1), C = tokens.extent(2);
  if (T != H * W || images.extent(0) != B || images.extent(1) != C ||
      images.extent(2) != H || images.extent(3) != W)
    throw std::invalid_argument("unpatchify_forward: dimension mismatch");
  if (B == 0 || T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, C}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t c) {
        images(b, c, t / W, t % W) = tokens(b, t, c);
      });
}

}  // namespace KokkosDNN

#endif
