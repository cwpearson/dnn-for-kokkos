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

.. cpp:function:: template <typename ExecSpace, typename Scalar> void add_bias_2d(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const Kokkos::View<Scalar*, ExecSpace>& b, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void add_bias_1d(const Kokkos::View<Scalar*, ExecSpace>& y, const Kokkos::View<Scalar*, ExecSpace>& b, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemm_bias(const DNNHandle<ExecSpace>& handle, Scalar alpha, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemv_bias(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

Dense layer — header: ``src/KokkosDNN_Dense.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dense_forward(const DNNHandle<ExecSpace>& handle, Scalar alpha, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dense_backward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dY, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dX, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dW, const Kokkos::View<Scalar*, ExecSpace>& db, const std::string& label)

Normalization and softmax
-------------------------

RMSNorm — header: ``src/KokkosDNN_RMSNorm.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_row(const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_noscale(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_row_noscale(const Kokkos::View<Scalar*, ExecSpace>& x, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

LayerNorm — header: ``src/KokkosDNN_LayerNorm.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar*, ExecSpace>& beta, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_row(const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar*, ExecSpace>& beta, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

Softmax — header: ``src/KokkosDNN_Softmax.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void softmax_rows(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void softmax_rows_causal(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

LayerScale — header: ``src/KokkosDNN_LayerScale.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layerscale_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Elementwise ``y = x * gamma`` (batched, single-sequence, and residual
   overloads exist; see the header).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layerscale_residual_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& residual, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

Attention, RoPE, and KV-cache
-----------------------------

Attention — header: ``src/KokkosDNN_Attention.hpp``. Streaming
online-softmax kernels (FlashAttention-style): one thread per (query
token, query head) streams over keys with O(1) extra memory.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_gqa_streaming(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, std::size_t q_off, const OView& out, float scale, std::size_t sliding_window, const std::string& label)

   Causal grouped-query attention over ``[T, H, D]``-style views, with
   optional sliding window. Dispatches to cuDNN SDPA when enabled and
   suitable, else the native kernel.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_gqa_single(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, const OView& out, float scale, const std::string& label)

   Single-query (decode) GQA step, plus a ``_sliding`` variant. Scalar-deduced
   overloads (``attention_gqa_single<ExecSpace>(...)``) also exist.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_bidirectional_streaming(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, const OView& out, float scale, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename VView, typename OView> void attention_mha_bidirectional(const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k, const VView& v, const OView& out, float scale, const std::string& label)

RoPE — header: ``src/KokkosDNN_RoPE.hpp``.

.. cpp:function:: template <typename ExecSpace> Kokkos::View<float*, ExecSpace> rope_inv_freq(int head_dim, float theta, const std::string& label = "")

   Llama-style inverse frequencies (plus ``rope_type``/NTK/YaRN overloads
   and ``rope_inv_freq_proportional`` for partial rotary factors).

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView, typename FView, typename OView> void rope_neoX_inplace(const QView& q, const KView& k, const FView& inv_freq, std::size_t pos, const std::string& label)

   In-place NeoX-style 1-D RoPE on ``[T, H, D]`` Q/K views.

.. cpp:function:: template <typename ExecSpace, typename Scalar, typename QView, typename KView> void rope2d_axial_inplace(const QView& q, const KView& k, const Kokkos::View<float*, ExecSpace>& inv_freq, int Hp, int Wp, const std::string& label)

   2-D axial RoPE for vision tokens (plus ``rope2d_axial_forward`` and the
   ``rope2d_inv_freq`` / ``rope2d_rotate_range`` helpers).

KV-cache — header: ``src/KokkosDNN_KVCache.hpp``.

Paged and quantized KV-cache writers (``kv_write_paged_dense``,
``kv_write_paged_quantized``, ``kv_write_quantized_contiguous``), prefix
copies (``copy_prefix_3`` / ``copy_prefix_4``), identity-table fill
(``fill_identity``), and paged decode entry points
(``attention_gqa_paged``, ``attention_gqa_paged_quantized``,
``attention_gqa_quantized``). See the header for the exact view layouts.

Embeddings, patches, and token assembly
---------------------------------------

Embedding — header: ``src/KokkosDNN_Embedding.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void embedding_gather(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table, const Kokkos::View<int*, ExecSpace>& ids, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void embedding_gather_scaled(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table, const Kokkos::View<int*, ExecSpace>& ids, float scale, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

Patch embedding — header: ``src/KokkosDNN_PatchEmbed.hpp``.

.. cpp:function:: void patchembed_grid(std::size_t H, std::size_t W, std::size_t K, std::size_t S, std::size_t& Hp, std::size_t& Wp, std::size_t& Np)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void patchembed_im2col(const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& patches, std::size_t Hp, std::size_t Wp, std::size_t K, std::size_t S, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void patchembed_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, std::size_t K, std::size_t S, const std::string& label)

Token assembly — header: ``src/KokkosDNN_Tokens.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void tokens_assemble_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cls, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& patches, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& regs, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

   Concatenates CLS / patch / register tokens into one sequence (plus
   segment-copy helpers ``tokens_copy_segment``,
   ``tokens_copy_cls_broadcast``, ``tokens_copy_regs_broadcast``).

Activations
-----------

SwiGLU — header: ``src/KokkosDNN_SwiGLU.hpp``.

.. cpp:function:: float silu(float x)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void swiglu(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

Pointwise activations — header: ``src/KokkosDNN_Activation.hpp``.

``gelu_forward`` (``GeluVariant::ExactErf`` / ``::Tanh``), ``silu`` via
SwiGLU, ``sigmoid_forward`` / ``sigmoid_backward``, ``tanh_forward`` /
``tanh_backward``, ``relu_forward`` / ``relu_backward``,
``logit_softcap_forward``, and ``gated_gelu_multiply``. All are
``template <class ExecSpace, class Scalar>`` over 2-D (and row-wise
1-D) views with a trailing label.

Training: losses and optimizers
-------------------------------

Losses — header: ``src/KokkosDNN_Loss.hpp``.

Cross-entropy (with logits) forward/backward and ``bce_with_logits``
forward/backward over ``[N, C]`` logits plus integer/float targets. See
the header for the exact reduction and weighting arguments.

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
   greedy decoding.

.. cpp:function:: template <typename ExecSpace, typename Scalar> int sample_token(const Kokkos::View<Scalar*, ExecSpace>& logits, const SampleOptions& opt, const std::vector<int>& prior, HostRNG& rng)

.. cpp:function:: template <typename ExecSpace, typename Scalar> int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits)

.. cpp:function:: template <typename ExecSpace, typename Scalar> int device_argmax(const Kokkos::View<Scalar*, ExecSpace>& logits)

   Host-side helpers (``softmax``, ``top_k_keep``, ``top_p_keep``,
   ``min_p_keep``, ``categorical``, ``apply_repetition_penalty``) back the
   sampler; see the header.

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
