Recurrence and audio DSP
=========================

LSTM — header: ``src/KokkosDNN_LSTM.hpp``. Single-layer bidirectional
LSTM, batch-first, with per-batch lengths (the StyleTTS2/Kokoro
recurrence stack: single-layer, ``batch_first=True``,
``bidirectional=True``, PyTorch ``i, f, g(cell), o`` gate order,
zero initial states). Lengths semantics match
``pack_padded_sequence(enforce_sorted=False)`` for the single-layer
case: steps ``t >= len[b]`` are skipped (states freeze) and outputs
there are zero. Scalar-agnostic I/O with FP32 accumulation; portable
team-parallel kernel, no backend ifdefs, runs on Serial.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void lstm_bidirectional(const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<int*, ExecSpace>& lengths, const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& w_ih, const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& w_hh, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& b_ih, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& b_hh, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   ``x [B, T, D]`` inputs, ``lengths [B]`` valid steps per batch,
   ``w_ih [2, 4H, D]`` / ``w_hh [2, 4H, H]`` weights (direction 0 =
   forward, 1 = reverse), ``b_ih/b_hh [2, 4H]`` biases, ``y [B, T,
   2H]`` forward/reverse-concatenated outputs. A rank-2 single-step
   overload also exists; see the header.

Audio DSP — header: ``src/KokkosDNN_AudioDSP.hpp``. Small FP32 audio
toolkit for vocoder-style models (Kokoro / ISTFTNet): Hann window,
STFT (magnitude/phase), inverse STFT, and 1D resampling with PyTorch
semantics. Portable kernels only.

.. cpp:function:: template <typename ExecSpace> void fill_hann_window(const Kokkos::View<float*, ExecSpace>& w, const std::string& label)

   Hann window, fftbins convention: ``w[n] = 0.5 - 0.5*cos(2*pi*n/N)``.

.. cpp:function:: template <typename ExecSpace> void stft_forward(const Kokkos::View<float*, ExecSpace>& x, const Kokkos::View<float*, ExecSpace>& win, size_t hop, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& mag, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& ph, const std::string& label)

   ``x [N]`` to ``mag/phase [F, W/2+1]`` with ``F = 1 + N/hop`` (N
   must divide evenly by hop). Center/reflect convention matching
   ``torch.stft`` (``center=True``, ``pad_mode='reflect'``,
   unnormalized, one-sided); forward DFT unscaled like torch.

.. cpp:function:: template <typename ExecSpace> void istft_inverse(const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& mag, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& ph, const Kokkos::View<float*, ExecSpace>& win, size_t hop, const Kokkos::View<float*, ExecSpace>& y, const std::string& label)

   Overlap-add synthesis matching ``torch.istft`` (``center=True``):
   per-frame IDFT (``1/W`` scaled), re-windowed, divided by the
   sum-of-squares envelope, center-trimmed to ``(F-1)*hop`` samples.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void upsample_nearest_1d(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, size_t scale, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Nearest-neighbor upsample over time-major ``[T, C]`` (integer
   ``scale`` repeat, matching ``F.interpolate(mode='nearest')``). A
   rank-1 overload also exists.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void interpolate_linear_1d(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Linear resample over time-major ``[T, C]`` with the PyTorch
   ``align_corners=False`` convention (both up and downsampling).
