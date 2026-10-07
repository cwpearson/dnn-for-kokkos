// Benchmark driver: initializes Kokkos around Google Benchmark.
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
