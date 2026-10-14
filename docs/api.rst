API reference
=============

All public entry points live in namespace ``KokkosDNN`` and are declared in
``src/KokkosDNN_*.hpp``. The umbrella header ``src/KokkosDNN.hpp`` pulls in
every core component (plus ``images/KokkosDNN_Image.hpp`` when
``KokkosDNN_ENABLE_IMAGE`` is on).

Conventions
-----------

* Every op is a function template on ``ExecSpace`` (any Kokkos execution
  space) and usually on ``Scalar`` (``float`` or
  ``Kokkos::Experimental::bhalf_t``). Accumulation inside kernels is FP32;
  results are cast back to ``Scalar``.
* 2-D data uses row-major views:
  ``Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>``.
* Most ops take a trailing ``label`` (kernel label for profiling).
* Ops that can dispatch to a CUDA library take
  ``const DNNHandle<ExecSpace>& handle`` as their first argument; on
  non-CUDA builds (or with the TPL disabled) the handle is ignored and the
  native kernel runs.
* Non-public helpers live in ``KokkosDNN::Impl`` and are not documented
  here; see the headers for tuning notes.

Handle and configuration
------------------------

Header: ``src/KokkosDNN_Handle.hpp``.

.. cpp:class:: template <typename ExecSpace> DNNHandle

   Records which third-party dispatches are active (``use_cublas``,
   ``use_cudnn``) and lazily caches the cuBLAS/cuDNN handles. Non-copyable,
   movable. Native Kokkos paths ignore it, so call sites stay free of
   ``#ifdef``\ s.

   .. cpp:function:: void* cublas_raw() const

      Lazily created cached cuBLAS handle, or ``nullptr`` when unusable
      (non-CUDA space, TPL disabled, creation failure). Callers fall back
      to the native kernel then.

   .. cpp:function:: void* cudnn_raw() const

      Lazily created cached cuDNN handle, same ``nullptr`` contract as
      above.

.. cpp:function:: void print_configuration(std::ostream& os)

   Prints the library configuration (enabled TPLs, Kokkos backend info).
   Declared in ``src/KokkosDNN.hpp``, defined in
   ``src/KokkosDNN_PrintConfiguration.cpp``.

Linear algebra
--------------

GEMM / GEMV — header: ``src/KokkosDNN_Gemm.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemm(const DNNHandle<ExecSpace>& handle, Scalar alpha, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, Scalar beta, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label = "KokkosDNN::gemm")

   ``Y = alpha * X * W^T + beta * Y`` with ``X : [M, K]``, ``W : [N, K]``
   (HuggingFace Linear layout), ``Y : [M, N]``. Dispatch ladder on CUDA:
   cuBLAS (``sgemm`` for ``float``, ``cublasGemmEx`` BF16 I/O with FP32
   accumulate for ``bhalf_t``) → shared-memory tiled native kernel →
   naive MDRange kernel on non-CUDA backends.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemv(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label = "KokkosDNN::gemv")

   Rank-1 decode variant: ``y[n] = sum_k x[k] * W[n, k]``. Shares
   :cpp:func:`gemm`'s dispatch ladder (used by the LM head).

Bias add and fused GEMM+bias — header: ``src/KokkosDNN_GemmBias.hpp``.
The rank-1/rank-2 bias adds are ``KokkosDNN::Impl`` helpers
(``Impl::add_bias_2d`` / ``Impl::add_bias_1d``); the public entry points are
the fused ops below. An empty bias view (extent 0) selects the unbiased path.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemm_bias(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

   ``Y = X * W^T + b`` (bias empty → unbiased). GEMM part reuses
   :cpp:func:`gemm`'s dispatch ladder.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemv_bias(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

Dense layer — header: ``src/KokkosDNN_Dense.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dense_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

   ``Y = X * W^T + b`` (bias is required here; use :cpp:func:`gemm_bias`
   for the optional-bias form). Forward reuses :cpp:func:`gemm`.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dense_backward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dY, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dX, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dW, const Kokkos::View<Scalar*, ExecSpace>& db, const std::string& label)

Normalization and softmax
-------------------------

RMSNorm — header: ``src/KokkosDNN_RMSNorm.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Rank-2 ``[T, D]`` form; a batched rank-3 ``[B, T, D]`` overload with the
   same per-row math also exists.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_row(const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_noscale(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_row_noscale(const Kokkos::View<Scalar*, ExecSpace>& x, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_one_plus(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& w, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Gemma2 variant: ``y = x / sqrt(mean(x^2) + eps) * (1 + w)`` (zero-init
   weights, HF ``Gemma2RMSNorm``).

LayerNorm — header: ``src/KokkosDNN_LayerNorm.hpp``. TeamPolicy kernels (one
team per normalized row).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar*, ExecSpace>& beta, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_row(const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar*, ExecSpace>& beta, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_noaffine_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Affine-free LayerNorm, rank-2 ``[T, D]`` form; a rank-3 ``[B, T, D]``
   overload with the same per-row math also exists.

Softmax — header: ``src/KokkosDNN_Softmax.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void softmax_rows(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void softmax_rows_causal(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

LayerScale — header: ``src/KokkosDNN_LayerScale.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layerscale_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Elementwise ``y = x * gamma`` (batched, single-sequence, and residual
   overloads exist; see the header).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layerscale_residual_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& residual, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Rank-1 ``[D]`` and batched rank-3 ``[B, T, D]`` overloads exist for both
   entry points; see the header.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void scale_by_scalar(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float scalar, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

   Uniform ``out = scalar * x`` (e.g. Gemma4 per-layer ``layer_scalar``);
   distinct from LayerScale's per-channel ``[D]`` multiply. Rank-1
   ``[D]`` overload also exists.

Diffusion conditioning — header: ``src/KokkosDNN_Diffusion.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void adaln_modulate_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& shift, const Kokkos::View<Scalar*, ExecSpace>& scale, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   AdaLN modulation ``y = x * (1 + scale) + shift``. Rank-3 ``[B, T, D]``
   overloads with ``[B, D]`` or ``[B, 1, D]`` shift/scale broadcast over
   tokens also exist.

.. cpp:function:: template <typename ExecSpace> void timestep_sinusoidal(const Kokkos::View<float*, ExecSpace>& t, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

   Sinusoidal timestep embedding ``t [N] -> out [N, F]`` (``F`` even;
   cos/sin halves). Frequencies are recomputed in-kernel, no temporaries.

Convolutions — headers: ``src/KokkosDNN_Conv.hpp``,
``src/KokkosDNN_DepthwiseConv.hpp``. Token-row ``[T, C]`` views with
``T = H*W`` row-major, stride 1, symmetric zero padding, FP32
accumulation. Bias is ``[Cout]``/``[C]`` or empty (extent 0) for no bias.
Native kernels only (no cuDNN dispatch; see the header's TPL-status note
for why). The ``handle`` is still taken for API uniformity.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void conv3x3_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H, size_t W, const std::string& label)

   Dense 3x3, pad 1: ``x [T, Cin]``, ``w [Cout, Cin, 3, 3]``,
   ``y [T, Cout]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void depthwise_conv_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t K, size_t H, size_t W, const std::string& label)

   Depthwise ``KxK`` (``K`` odd), pad ``K/2``: ``x/y [T, C]``,
   ``w [C, K, K]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void grouped_conv1x1_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t groups, const std::string& label)

   Grouped 1x1, no padding: ``x [T, Cin]``, ``w [Cout, Cg]``
   (``Cg = Cin / groups``), ``y [T, Cout]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void depthwise_conv3x3_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H, size_t W, const std::string& label)

   Depthwise 3x3 convenience op (header
   ``src/KokkosDNN_DepthwiseConv.hpp``, no handle): ``x/y [T, C]``,
   ``w [C, 3, 3]``; a batched ``[B, T, C]`` overload also exists.

Attention, RoPE, and KV-cache
-----------------------------

Attention — header: ``src/KokkosDNN_Attention.hpp``. Streaming
online-softmax kernels (FlashAttention-style): one thread per (query
token, query head) streams over keys with O(1) extra memory. The
``Impl::attention_*_streaming`` cores are not public; the entry points are
the ``attention_*_single`` / ``attention_*_bidirectional`` overloads below
(each with an explicit ``<ExecSpace, Scalar>`` overload plus a
Scalar-deduced overload).

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_gqa_single(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, std::size_t q_off, const OView& out, float scale, const std::string& label)

   Causal grouped-query attention over ``[T, H, D]``-style views
   (``q [T, Hq, D]``, ``k/v [Ktot, Hkv, D]``, ``out [T, Hq, D]``;
   query ``t`` attends keys ``[0, q_off + t]``). Dispatches to cuDNN SDPA
   when enabled and suitable, else the native kernel.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_gqa_single_sliding(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, std::size_t q_off, const OView& out, float scale, std::size_t sliding_window, const std::string& label)

   Sliding-window causal GQA (e.g. Gemma4 sliding layers): query ``t`` at
   absolute position ``p = q_off + t`` attends keys
   ``[max(0, p + 1 - sliding_window), p]``. ``sliding_window == 0``
   degrades to full causal attention.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename MView, typename OView> void attention_gqa_single_softcap(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, const MView& key_valid, std::size_t q_off, const OView& out, float scale, float softcap, const std::string& label)

   Causal GQA with Gemma2 tanh logit softcapping
   (``softcap * tanh(q.k * scale / softcap)``, ``softcap <= 0`` disables)
   and a per-key validity mask (``key_valid [Ktot]``). Native kernel only
   (cuDNN frontend has no softcap/mask knob); see the header.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_mha_bidirectional(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, const OView& out, float scale, const std::string& label)

   Bidirectional (non-causal) attention over ``[T, H, D]``-style views
   (every query attends all keys; ``scale <= 0`` selects ``1/sqrt(D)``).
   Dispatches to cuDNN SDPA when enabled and suitable, else native.

RoPE — header: ``src/KokkosDNN_RoPE.hpp``.

.. cpp:function:: template <typename ExecSpace> Kokkos::View<float*, ExecSpace> rope_inv_freq(int head_dim, float theta, const std::string& label = "")

   Llama-style inverse frequencies. A ``rope_type``/NTK/YaRN overload
   ``rope_inv_freq(head_dim, theta, rope_type, factor, low_freq,
   high_freq, original_max)`` (``"llama3"`` scaling) and
   ``rope_inv_freq_proportional(head_dim, theta, partial_rotary_factor)``
   for partial rotary factors (trailing dims pass through) also exist.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename FView, typename PView> void rope_neoX_inplace(const QView& q, const KView& k, const FView& inv_freq, int head_dim, const PView& positions, const std::string& label)

   In-place NeoX-style 1-D RoPE on ``[T, H, D]`` Q/K views with per-token
   ``positions [T]``. Layout-agnostic; empty ``k`` rotates ``q`` only.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView> void rope2d_axial_inplace(const QView& q, const KView& k, const Kokkos::View<float*, ExecSpace>& inv_freq, int Hp, int Wp, size_t prefix_skip, const std::string& label)

   2-D axial RoPE for vision tokens (``prefix_skip`` untouched prefix rows,
   default 5 = 1 CLS + 4 registers). Plus ``rope2d_axial_forward`` (out-of-place)
   and the ``rope2d_inv_freq`` host/device helpers (``head_dim/4``
   entries, ``theta = 100`` default).

KV-cache — header: ``src/KokkosDNN_KVCache.hpp``.

Paged and quantized KV-cache writers (``kv_write_paged_dense``,
``kv_write_paged_quantized``, ``kv_write_quantized_contiguous``) and paged
decode entry points (``attention_gqa_paged``,
``attention_gqa_paged_quantized``, ``attention_gqa_quantized``). Pool-growth
helpers (``Impl::copy_prefix_3`` / ``Impl::copy_prefix_4``,
``Impl::fill_identity``) are implementation details. Paged/quantized paths
run the native streaming kernel (no cuDNN dispatch). See the header for the
exact view layouts.

Embeddings, patches, and token assembly
---------------------------------------

Embedding — header: ``src/KokkosDNN_Embedding.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void embedding_gather(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table, const Kokkos::View<int*, ExecSpace>& ids, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void embedding_gather_scaled(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table, const Kokkos::View<int*, ExecSpace>& ids, float scale, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

Patch embedding — header: ``src/KokkosDNN_PatchEmbed.hpp``.
ViT patch-embed as im2col + biased GEMM (``Impl::patchembed_grid`` /
``Impl::patchembed_im2col`` are helpers, not public entry points).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void patchembed_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& weight, const Kokkos::View<Scalar*, ExecSpace>& bias, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& output, int K, int S, const std::string& label)

   ``images [B, C, H, W]`` → ``output [B, Np, D]`` with
   ``weight [D, C*K*K]`` and ``bias [D]`` (empty = unbiased). GEMM reuses
   :cpp:func:`gemm_bias`'s dispatch ladder.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void unpatchify_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& tokens, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& image, size_t H, size_t W, const std::string& label)

   Inverse patch embed: ``tokens [T, C]`` → ``image [C, H, W]``
   (``T = H*W`` row-major transpose). A batched ``[B, T, C]`` →
   ``[B, C, H, W]`` overload also exists.

Token assembly — header: ``src/KokkosDNN_Tokens.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void tokens_assemble_forward(const Kokkos::View<Scalar*, ExecSpace>& cls, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& regs, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& patches, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

   Concatenates CLS / patch / register tokens into one ``[B, T, D]``
   sequence (``T = 1 + R + Np``). Fully-batched ``[B, 1, D]``-CLS and
   ``[1, D]``-CLS overloads also exist, plus the single-sequence
   ``tokens_assemble_single`` (``[D]``/``[1, D]`` CLS → ``[T, D]``)
   convenience forms. Segment-copy helpers (``Impl::tokens_copy_*``)
   are implementation details.

Activations
-----------

SwiGLU — header: ``src/KokkosDNN_SwiGLU.hpp``.

.. cpp:function:: float silu(float x)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void swiglu(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

Pointwise activations — header: ``src/KokkosDNN_Activation.hpp``.

``gelu_forward`` (``GeluVariant::ExactErf`` / ``::Tanh``),
``silu_forward``, ``sigmoid_forward`` / ``sigmoid_backward``,
``tanh_forward`` / ``tanh_backward``, ``relu_forward`` /
``relu_backward``, ``logit_softcap_forward``, and ``gated_gelu_multiply``. All are
``template <class ExecSpace, class Scalar>`` over 2-D (and row-wise
1-D) views with a trailing label. ``silu_forward`` computes ``silu``
elementwise (see ``src/KokkosDNN_SwiGLU.hpp``); ``gated_gelu_multiply``
defaults to ``GeluVariant::Tanh`` (Gemma MLP).

Training: losses and optimizers
-------------------------------

Losses — header: ``src/KokkosDNN_Loss.hpp``. Binary cross-entropy with
logits over ``[B, Out]`` logits/targets (XOR dense outputs feed in
directly), FP32, mean reduction.

.. cpp:function:: template <typename ExecSpace, typename Scalar> float bce_with_logits_loss(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& targets, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void bce_with_logits_backward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& targets, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dlogits, const std::string& label)

Optimizers — header: ``src/KokkosDNN_Optimizer.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void sgd_step(const Kokkos::View<Scalar*, ExecSpace>& param, const Kokkos::View<Scalar*, ExecSpace>& grad, float lr, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void adam_step(const Kokkos::View<Scalar*, ExecSpace>& param, const Kokkos::View<Scalar*, ExecSpace>& grad, const Kokkos::View<float*, ExecSpace>& m, const Kokkos::View<float*, ExecSpace>& v, float lr, float beta1, float beta2, float eps, float t, const std::string& label)

   SGD (plain + momentum overloads) and Adam (bias-corrected, FP32
   moments). Rank-1 and rank-2 overloads exist; see the header.

Sampling
--------

Header: ``src/KokkosDNN_Sampler.hpp``.

.. cpp:struct:: SampleOptions

   Temperature / top-k / top-p / min-p, repetition penalty, and seed.
   :cpp:func:`is_greedy_sample` reports whether the options reduce to
   greedy decoding. ``SampleOptions::llama_3_1_8b_instruct()`` presets
   Meta's recommended Llama 3.1 Instruct values (temperature 0.6,
   top-p 0.9).

.. cpp:function:: template <typename ExecSpace, typename Scalar> int sample_token(const Kokkos::View<Scalar*, ExecSpace>& logits, const SampleOptions& opt, const std::vector<int>& prior, HostRNG& rng)

.. cpp:function:: template <typename ExecSpace, typename Scalar> int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits)

.. cpp:function:: template <typename ExecSpace, typename Scalar> int device_argmax(const Kokkos::View<Scalar*, ExecSpace>& logits)

   Host-side pipeline (``apply_repetition_penalty``, ``top_k_keep``,
   ``top_p_keep``, ``min_p_keep``, ``sample_host_logits`` over float
   logits plus ``Impl`` softmax/categorical helpers) backs the sampler;
   see the header. Greedy sampling without a repetition penalty stays
   on-device via :cpp:func:`device_argmax`.

Optional image component
------------------------

Headers: ``src/images/KokkosDNN_Image*.hpp``. Built only with
``KokkosDNN_ENABLE_IMAGE=ON`` (stb-backed JPEG/PNG/BMP decode plus
resize/normalize preprocessing into device views). See
``src/images/KokkosDNN_ImageIO.hpp``,
``src/images/KokkosDNN_ImagePreprocess.hpp``, and
``src/images/KokkosDNN_ImageTypes.hpp``.

TPL dispatch notes
------------------

* ``src/tpl/cublas/`` — GEMM dispatch to cuBLAS; handle lifecycle in
  ``KokkosDNN_CuBLAS_Handle.hpp``.
* ``src/tpl/cudnn/`` — attention dispatch to cuDNN SDPA graph API with
  automatic native fallback; handle lifecycle in
  ``KokkosDNN_CuDNN_Handle.hpp``.
