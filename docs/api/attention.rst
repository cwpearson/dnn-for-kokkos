Attention, RoPE, and KV-cache
==============================

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
