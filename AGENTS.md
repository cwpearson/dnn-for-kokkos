# Commit guidelines

LLM-authored commits must include a git `Co-authored-by` trailer naming the model.
Example for Muse Spark 1.3:
`Co-authored-by: Muse Spark 1.3 <noreply@carlpearson.net>`

# Project layout

```text
src/             core library (native Kokkos ops)
src/tpl/         optional CUDA TPL dispatch (cublas, cudnn)
examples/llama/  Llama inference example (not part of the library)
examples/xor/    XOR MLP training example
tests/           core op unit tests
benchmarks/      microbenchmarks (native vs TPL dispatch)
cmake/           config header + package config templates
```

# Design Guidelines

- Non-public functions / APIs should be in `KokkosDNN::Impl` namespace.
- Non-public macros should be prefixed `KOKKOSDNN_IMPL_`
