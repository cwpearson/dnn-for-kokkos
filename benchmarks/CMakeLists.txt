# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Google Benchmark-based microbenchmarks for core KokkosDNN operators.
#
# Produces a single `kokkosdnn_bench` binary. GEMM, biased-linear, and
# attention register Native vs Dispatched variants so native Kokkos kernels
# can be compared against TPL paths (cuBLAS GEMM incl. gemm_bias/gemv_bias/
# dense_forward reuse, cuDNN causal/sliding/bidirectional SDPA) on CUDA builds:
#
#   ./kokkosdnn_bench --benchmark_filter='Gemm.*'
#   ./kokkosdnn_bench --benchmark_filter='.*Cublas.*|.*Native.*'
#
# When KokkosDNN_ENABLE_CUBLAS/CUDNN are OFF (or on non-CUDA spaces) the
# Dispatched benchmarks fall back to the native kernel, so Native ==
# Dispatched by construction. That keeps the binary useful on CPU-only CI.

include(FetchContent)

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/v1.9.0.zip
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  # Third-party policy: no -Werror in benchmark's own build (its
  # unconditional -pedantic-errors trips hipcc on __COUNTER__).
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
endif()
# Third-party policy: benchmark builds warning-free (its own -w on every
# compiled target) and its headers reach kokkosdnn_bench as SYSTEM, so they
# can never trip -Werror. Suppress both libs (benchmark+benchmark_main) in
# both spellings (system packages provide benchmark::benchmark as IMPORTED).
kokkosdnn_suppress_third_party_warnings(benchmark)
kokkosdnn_suppress_third_party_warnings(benchmark_main)
kokkosdnn_suppress_third_party_warnings(benchmark::benchmark)
kokkosdnn_suppress_third_party_warnings(benchmark::benchmark_main)

# Standalone bandwidth benchmark for the packed-quant GEMV (decode) path: reads
# the real matrices of a GGUF checkpoint and reports GB/s per shape, which is
# what the decode loop is bound by. Separate binary because it needs a model
# file and has its own main() (no google-benchmark fixture):
#   ./kokkosdnn_bench_gguf_gemv <model.gguf> [iters] [rows] [tpr] [groups]
# rows/tpr/groups (default 0 = the shipped layout) drive the row-blocked kernel
# directly, for tuning the layout against a real checkpoint.
#   ./kokkosdnn_bench_gguf_gemv <model.gguf> 40 2 64 2 blk.0.ffn_gate.weight
add_executable(kokkosdnn_bench_gguf_gemv Bench_GGUFGemv.cpp)
target_link_libraries(kokkosdnn_bench_gguf_gemv PRIVATE KokkosDNN::kokkosdnn)
set_target_properties(kokkosdnn_bench_gguf_gemv PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_bench_gguf_gemv)

add_executable(kokkosdnn_bench
  BenchmarkMain.cpp
  Bench_Gemm.cpp
  Bench_GemmBias.cpp
  Bench_Attention.cpp
  Bench_AttentionSliding.cpp
  Bench_AttentionBidirectional.cpp
  Bench_DecodeAttention.cpp
  Bench_MoeDecode.cpp
  Bench_Elementwise.cpp
  Bench_Sampler.cpp
)
target_link_libraries(kokkosdnn_bench PRIVATE KokkosDNN::kokkosdnn benchmark::benchmark)
set_target_properties(kokkosdnn_bench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_bench)
# BENCHMARK() pastes __COUNTER__ into this TU; Clang-family compilers flag
# it as a C2y extension under -pedantic. Probe-and-disable just that
# warning for this target (GCC never warns here, MSVC has no equivalent).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-c2y-extensions KOKKOSDNN_HAS_WNO_C2Y_EXTENSIONS)
if(KOKKOSDNN_HAS_WNO_C2Y_EXTENSIONS)
  target_compile_options(kokkosdnn_bench PRIVATE -Wno-c2y-extensions)
endif()

# Smoke entry for ctest: list benchmarks (fast, exercises Kokkos init).
include(CTest)
add_test(NAME kokkosdnn_bench_list COMMAND kokkosdnn_bench --benchmark_list_tests)
