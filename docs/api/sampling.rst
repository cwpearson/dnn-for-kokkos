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

.. cpp:function:: template <typename ExecSpace, typename Scalar> SampleResult sample_token_result(const Kokkos::View<Scalar*, ExecSpace>& logits, const SampleOptions& opt, const std::vector<int>& prior, HostRNG& rng)

.. cpp:function:: template <typename ExecSpace, typename Scalar> int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits)

.. cpp:function:: template <typename ExecSpace, typename Scalar> int device_argmax(const Kokkos::View<Scalar*, ExecSpace>& logits)

.. cpp:class:: template <typename ExecSpace> Impl::DeviceSampler

   Device-side sampling pipeline used by ``sample_token``/``sample_token_result``:
   penalties and logit bias, temperature scaling, top-k/top-p/min-p
   truncation and the categorical draw all run as device kernels, and only
   the sampled id (4 bytes) is copied back. A top-N logprob request adds a
   small candidate gather; the requested entries are then selected on the
   host. Truncation thresholds are resolved on a log-probability grid
   refined at the binding boundary (see the header for the documented
   distribution-level differences from the host reference).

   ``DeviceSampler`` owns V-sized scratch, so a decode loop should hold one
   for the whole request (``Session`` and ``Model::generate`` do) rather than
   calling the free functions per token. Greedy sampling with neutral
   penalties/bias and no logprobs stays on-device via
   :cpp:func:`device_argmax` and never touches the scratch.

   Host-side helpers (``apply_repetition_penalty``, ``apply_logit_bias``,
   ``apply_frequency_presence``, ``top_k_keep``, ``top_p_keep``,
   ``min_p_keep``, ``sample_host_logits`` /
   ``sample_host_logits_result`` plus the ``Impl`` softmax/categorical
   helpers) remain available and define the reference semantics the device
   pipeline reproduces.
