// GEMM: Y = X * W^T vs host reference.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    KokkosDNN::DNNHandle<Exec> h;
    const int M = 5, K = 7, N = 4;
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
        Y("Y", M, N);
    auto hx = Kokkos::create_mirror_view(X);
    auto hw = Kokkos::create_mirror_view(W);
    auto hy = Kokkos::create_mirror_view(Y);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k) hx(i, k) = 0.1f * (i * K + k + 1);
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k) hw(j, k) = 0.01f * (j * K - k + 3);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) hy(i, j) = 1.0f;
    Kokkos::deep_copy(X, hx);
    Kokkos::deep_copy(W, hw);
    Kokkos::deep_copy(Y, hy);
    KokkosDNN::gemm(h, 2.0f, X, W, 1.0f, Y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(Y);
    Kokkos::deep_copy(got, Y);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float acc = 0;
        for (int k = 0; k < K; ++k) acc += hx(i, k) * hw(j, k);
        float want = 2 * acc + 1;
        if (std::fabs(got(i, j) - want) > 1e-4f) {
          std::printf("FAIL gemm (%d,%d) got %f want %f\n", i, j, got(i, j),
                      want);
          return 1;
        }
      }
    std::printf("PASS test_gemm\n");
  }
  Kokkos::finalize();
  return 0;
}
