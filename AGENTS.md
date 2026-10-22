# Commit guidelines

LLM-authored commits must include a git `Co-authored-by` trailer naming the model.
Example for Muse Spark 1.3:
`Co-authored-by: Muse Spark 1.3 <noreply@carlpearson.net>`

# Project layout

```text
src/             core library (native Kokkos ops; no model-specific code)
src/tpl/         optional CUDA TPL dispatch (cublas, cudnn)
src/images/      optional image component (stb-backed I/O + preprocessing)
models/          installable per-model components (KokkosDNN::lfm25, KokkosDNN::moss)
models/common/   shared model-support scaffold (JSON/BPE/chat/sampling/detok; header-only)
models/lfm2_5/   LFM2.5 model library (weights/tokenizer/config)
models/moss_image/ MOSS-Image pipeline headers (DiT/encoder/VAE/Qwen tokenizer)
server/          OpenAI-compatible HTTP server stub (thin binary; no model logic)
examples/        thin drivers linking models/* or core (lfm2_5, moss_image, llama, ...)
examples/llama/  Llama inference example (not part of the library)
examples/smollm2/ SmolLM2 inference example (not part of the library)
examples/xor/    XOR MLP training example
examples/dinov3/ DINOv3 image-encoder example (not part of the library)
examples/kokoro/  Kokoro TTS example (not part of the library)
tests/           core op unit tests
benchmarks/      microbenchmarks (native vs TPL dispatch)
cmake/           config header + package config templates
```

# Build Guidelines

- All builds must be clean with `-DKokkosDNN_ENABLE_COMPILER_WARNINGS=ON`
  (warning set mirroring `Kokkos_ENABLE_COMPILER_WARNINGS`, plus `-Werror`).
  Every first-party target gets the flags via
  `kokkosdnn_enable_compiler_warnings(<target>)`
  (`cmake/KokkosDNNCompilerWarnings.cmake`); call it for each new target.
  CI enables the option on every workflow, so a warning introduced anywhere
  (library, tests, examples, benchmarks, server) fails CI.

# Design Guidelines

- Follow the Kokkos performance portability ethos. Unless necessary for some reason, implementations should not be gated on specific Kokkos backends or execution space types.
- Prefer using existing Kokkos features rather than creating new equivalent utilities.
- Non-public functions / APIs should be in and `Impl` namespace (e.g. `KokkosDNN::Impl`)
- Non-public macros should be prefixed `KOKKOSDNN_IMPL_`
- Tests do not exist to freeze configuration decisions (e.g. default values) or implementation details, but to confirm that public functionality is correct.
