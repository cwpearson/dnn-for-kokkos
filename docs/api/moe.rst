Mixture-of-Experts and quantized checkpoints
==============================================

MoE routing — header: ``src/KokkosDNN_MoE.hpp``. Sparse
Mixture-of-Experts gating: per-token top-k expert select plus
combination weights. Two routing activations are supported because
model families gate differently (softmax for Qwen3-style routers,
sigmoid for LFM2-MoE-style routers); the expert FFNs themselves reuse
the existing :cpp:func:`gemm` / ``swiglu`` ops, so no dedicated
expert-GEMM kernel lives here. Scalar-agnostic, FP32 accumulation,
portable RangePolicy/MDRange kernels (the ``T=1`` decode path is the
same kernel with one iteration, so Serial/CPU execution spaces work).
``expert_bias`` is ``[E]`` or empty (extent 0) for no bias;
``top_idx`` is ``[T, top_k]`` expert ids (descending score) and
``top_w`` is ``[T, top_k]`` float weights.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void moe_route(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits, const Kokkos::View<Scalar*, ExecSpace>& expert_bias, int top_k, bool norm_topk, float routed_scale, const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& top_idx, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w, const std::string& label)

   Softmax routing over ``logits [T, E]`` (bias, when present, shifts
   the scores inside the softmax). ``norm_topk=true`` renormalizes the
   softmax over the selected ``K`` (sums to 1); ``false`` gathers a
   global ``E``-way softmax at the selected ids (sums to <= 1).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void moe_route_sigmoid(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits, const Kokkos::View<Scalar*, ExecSpace>& expert_bias, int top_k, bool norm_topk, float routed_scale, const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& top_idx, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w, const std::string& label)

   Sigmoid routing (LFM2-MoE-style): the bias, when present, shifts the
   sigmoid outputs for top-k *selection* only and is not part of the
   combination weights. ``norm_topk=true`` renormalizes the gathered
   sigmoids over the ``K`` selected experts (HF 1e-6 epsilon in the
   denominator); ``false`` keeps the raw gathered sigmoids.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void moe_expert_weighted_sum(const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& expert_out, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   ``y[t,d] = sum_k top_w[t,k] * expert_out[t,k,d]`` over stacked
   per-(token, expert) outputs ``[T, K, D]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void moe_gather_rows(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src, const Kokkos::View<int*, ExecSpace>& rows, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst, const std::string& label)

   Sparse dispatch gather: ``dst[n,d] = src[rows[n],d]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void moe_scatter_add_weighted(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src, const Kokkos::View<int*, ExecSpace>& rows, const Kokkos::View<float*, ExecSpace>& weights, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst, const std::string& label)

   Sparse combine: ``dst[rows[n],d] += weights[n] * src[n,d]`` (FP32
   accumulation; ``rows`` must hold unique entries, true for top-k
   dispatch, so no atomics are needed).

GGUF checkpoints — header: ``src/KokkosDNN_GGUF.hpp`` (namespace
``KokkosDNN::Gguf``). GGUFv3 container reader (llama.cpp-style files),
host-side dequantization to FP32, and device-resident quantized
GEMV/GEMM against packed weights with on-the-fly dequantization and
FP32 accumulation. Computable dtypes: F32/F16/BF16 plus Q8_0, Q4_K,
Q6_K (other dtypes throw); portable TeamPolicy kernels, no backend
ifdefs.

.. cpp:class:: GgufFile

   Parsed GGUFv3 file: ``GgufFile(path)`` memory-maps the file,
   exposes the tensor infos (name, ggml-order dims, dtype, offset)
   and string metadata, and hands out raw payload pointers for
   :cpp:func:`dequant_tensor_to_float` /
   :cpp:func:`load_quant_weight`.

.. cpp:function:: void dequant_tensor_to_float(const GgufFile& file, const GgufTensorInfo& info, std::vector<float>& out)

   Host-side dequantization of one tensor to FP32 (covers the
   computable dtype set above).

.. cpp:function:: template <typename ExecSpace> QuantWeight<ExecSpace> load_quant_weight(const GgufFile& file, const GgufTensorInfo& info)

   Upload of one tensor's packed payload into a device-resident
   ``QuantWeight`` (dtype, rows, cols, packed bytes).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void quant_gemv(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar*, ExecSpace>& x, const QuantWeight<ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

   ``y[N] = W[N,K] * x[K]`` with packed-quant ``W`` (native kernel
   only; the handle is kept for API symmetry with
   :cpp:func:`gemv`). An ``ExecSpace`` overload without a handle also
   exists.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void quant_gemm(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const QuantWeight<ExecSpace>& w, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

   ``Y[M,N] = X[M,K] * W[N,K]^T`` with packed-quant ``W`` (native
   kernel only). An ``ExecSpace`` overload without a handle also
   exists.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dequant_to_dense(const QuantWeight<ExecSpace>& w, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

   One-shot dequantization of a packed weight to a dense ``[rows,
   cols]`` matrix, so steady-state callers (e.g. MoE prefill over
   thin gathered slices) can run the dense :cpp:func:`gemm` dispatch
   instead of re-dequantizing per output row.
