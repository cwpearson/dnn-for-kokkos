# Google Benchmark-based microbenchmarks for core KokkosDNN operators.
#
# Produces a single `kokkosdnn_bench` binary. GEMM and attention register
# Native vs Dispatched variants so native Kokkos kernels can be compared
# against TPL paths (cuBLAS GEMM, cuDNN SDPA) on CUDA builds:
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
  FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(kokkosdnn_bench
  BenchmarkMain.cpp
  Bench_Gemm.cpp
  Bench_Attention.cpp
  Bench_Elementwise.cpp
)
target_link_libraries(kokkosdnn_bench PRIVATE KokkosDNN::kokkosdnn benchmark::benchmark)
set_target_properties(kokkosdnn_bench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# Smoke entry for ctest: list benchmarks (fast, exercises Kokkos init).
include(CTest)
add_test(NAME kokkosdnn_bench_list COMMAND kokkosdnn_bench --benchmark_list_tests)
