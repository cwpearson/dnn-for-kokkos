Sampling
========

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
