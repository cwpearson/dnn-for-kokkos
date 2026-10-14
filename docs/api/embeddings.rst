Embeddings, patches, and token assembly
==========================================

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
