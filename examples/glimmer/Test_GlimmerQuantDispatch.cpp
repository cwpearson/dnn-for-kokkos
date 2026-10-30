// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Dispatch-seam tests for KokkosDNN_GlimmerQuantDispatch.hpp: dense overloads
// must match the direct gemm/gemv/embedding_gather calls they replace, and
// the packed (QuantWeight) overloads must match dense references. The packed
// path is exercised with synthetic F32 payloads (exact packing), which
// instantiates every quant template the dense Model build otherwise leaves
// uninstantiated. Small shapes keep this backend-agnostic (Serial/CUDA).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "KokkosDNN_GlimmerQuantDispatch.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using KokkosDNN::Glimmer::Impl::embed_forward;
using KokkosDNN::Glimmer::Impl::linear_chunk;
using KokkosDNN::Glimmer::Impl::linear_decode;
using KokkosDNN::Glimmer::Impl::linear_prefill;
using KokkosDNN::Glimmer::Impl::lm_head_forward;
using Mat = Kokkos::View<float**, Kokkos::LayoutRight, Exec>;
using Vec = Kokkos::View<float*, Exec>;
using Row = Kokkos::View<float*, Exec>;

// Deterministic small-magnitude pattern (mix of signs).
float pattern(size_t i) {
  return (float)(((i * 7u) % 11u) - 5u) * 0.25f;
}

Mat make_mat(size_t R, size_t C, size_t base) {
  Mat m("m", R, C);
  auto h = Kokkos::create_mirror_view(m);
  for (size_t r = 0; r < R; ++r)
    for (size_t c = 0; c < C; ++c) h(r, c) = pattern(base + r * C + c);
  Kokkos::deep_copy(m, h);
  return m;
}

Vec make_vec(size_t N, size_t base) {
  Vec v("v", N);
  auto h = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < N; ++i) h(i) = pattern(base + i);
  Kokkos::deep_copy(v, h);
  return v;
}

Kokkos::View<int*, Exec> make_toks(const std::vector<int>& ids) {
  Kokkos::View<int*, Exec> t("t", ids.size());
  auto h = Kokkos::create_mirror_view(t);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(t, h);
  return t;
}

// Exact F32 packing of a dense reference: payload is raw row-major floats.
KokkosDNN::Gguf::QuantWeight<Exec> pack_f32(size_t R, size_t C, size_t base) {
  KokkosDNN::Gguf::QuantWeight<Exec> w;
  w.dtype = KokkosDNN::Gguf::GgmlType::F32;
  w.rows  = R;
  w.cols  = C;
  w.data  = Kokkos::View<uint8_t*, Exec>("qw", R * C * sizeof(float));
  std::vector<float> vals(R * C);
  for (size_t i = 0; i < vals.size(); ++i) vals[i] = pattern(base + i);
  auto h = Kokkos::create_mirror_view(w.data);
  std::memcpy(h.data(), vals.data(), vals.size() * sizeof(float));
  Kokkos::deep_copy(w.data, h);
  return w;
}

Mat dense_from_base(size_t R, size_t C, size_t base) {
  return make_mat(R, C, base);
}

std::vector<float> to_host(const Mat& m) {
  auto h = Kokkos::create_mirror_view(m);
  Kokkos::deep_copy(h, m);
  std::vector<float> out(m.extent(0) * m.extent(1));
  for (size_t r = 0; r < m.extent(0); ++r)
    for (size_t c = 0; c < m.extent(1); ++c)
      out[r * m.extent(1) + c] = h(r, c);
  return out;
}

std::vector<float> to_host(const Vec& v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return std::vector<float>(h.data(), h.data() + h.extent(0));
}

void expect_near_vec(const std::vector<float>& a, const std::vector<float>& b,
                     float tol) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], tol) << "i=" << i;
}

}  // namespace

TEST(GlimmerQuantDispatch, DensePrefillMatchesGemm) {
  KokkosDNN::DNNHandle<Exec> handle;
  const size_t M = 4, K = 16, N = 8;
  Mat X = make_mat(M, K, 0), W = make_mat(N, K, 100), Y("y", M, N),
      ref("ref", M, N), staging;
  KokkosDNN::gemm<Exec, float>(handle, 1.0f, X, W, 0.0f, ref, "ref");
  linear_prefill(handle, X, W, staging, Y, "t");
  expect_near_vec(to_host(Y), to_host(ref), 1e-6f);
}

TEST(GlimmerQuantDispatch, DenseDecodeMatchesGemm) {
  KokkosDNN::DNNHandle<Exec> handle;
  const size_t K = 16, N = 8;
  Mat X = make_mat(1, K, 0), W = make_mat(N, K, 100), Y("y", 1, N),
      ref("ref", 1, N), staging;
  KokkosDNN::gemm<Exec, float>(handle, 1.0f, X, W, 0.0f, ref, "ref");
  linear_decode(handle, X, W, staging, Y, "t");
  expect_near_vec(to_host(Y), to_host(ref), 1e-6f);
  // Chunk dispatcher picks decode for single-row X.
  Mat Y2("y2", 1, N);
  linear_chunk(handle, X, W, staging, Y2, "t");
  expect_near_vec(to_host(Y2), to_host(ref), 1e-6f);
  EXPECT_THROW(linear_decode(handle, make_mat(2, K, 0), W, staging, Y, "t"),
               std::invalid_argument);
}

TEST(GlimmerQuantDispatch, QuantPrefillMatchesDense) {
  KokkosDNN::DNNHandle<Exec> handle;
  const size_t M = 4, K = 16, N = 8;
  Mat X = make_mat(M, K, 0), Wd = dense_from_base(N, K, 100),
      Y("y", M, N), ref("ref", M, N), staging("st", N, K);
  auto Wq = pack_f32(N, K, 100);
  KokkosDNN::gemm<Exec, float>(handle, 1.0f, X, Wd, 0.0f, ref, "ref");
  // Dequant-once path uses the SAME gemm on bit-identical input.
  linear_prefill(handle, X, Wq, staging, Y, "t");
  expect_near_vec(to_host(Y), to_host(ref), 1e-6f);
  // Chunk dispatcher picks prefill for M > 1.
  Mat Y2("y2", M, N);
  linear_chunk(handle, X, Wq, staging, Y2, "t");
  expect_near_vec(to_host(Y2), to_host(ref), 1e-6f);
  // Staging must be exactly [N, K] (cuBLAS pitch contract).
  Mat bad("bad", N + 1, K);
  EXPECT_THROW(linear_prefill(handle, X, Wq, bad, Y, "t"),
               std::invalid_argument);
}

TEST(GlimmerQuantDispatch, QuantDecodeMatchesDense) {
  KokkosDNN::DNNHandle<Exec> handle;
  const size_t K = 16, N = 8;
  Mat X = make_mat(1, K, 0), Wd = dense_from_base(N, K, 100), Y("y", 1, N),
      ref("ref", 1, N), staging;
  auto Wq = pack_f32(N, K, 100);
  KokkosDNN::gemm<Exec, float>(handle, 1.0f, X, Wd, 0.0f, ref, "ref");
  linear_decode(handle, X, Wq, staging, Y, "t");
  // FP32 accumulation both sides; reduction order may differ.
  expect_near_vec(to_host(Y), to_host(ref), 1e-4f);
  Mat Y2("y2", 1, N);
  linear_chunk(handle, X, Wq, staging, Y2, "t");
  expect_near_vec(to_host(Y2), to_host(ref), 1e-4f);
  EXPECT_THROW(linear_decode(handle, make_mat(2, K, 0), Wq, staging, Y, "t"),
               std::invalid_argument);
}

TEST(GlimmerQuantDispatch, LmHeadDenseAndQuant) {
  KokkosDNN::DNNHandle<Exec> handle;
  const size_t K = 16, N = 8;
  Vec x = make_vec(K, 0), y("y", N), ref("ref", N);
  Mat Wd = dense_from_base(N, K, 100);
  auto Wq = pack_f32(N, K, 100);
  KokkosDNN::gemv<Exec, float>(handle, x, Wd, ref, "ref");
  lm_head_forward(handle, x, Wd, y, "t");
  expect_near_vec(to_host(y), to_host(ref), 1e-6f);
  Vec yq("yq", N);
  lm_head_forward(handle, x, Wq, yq, "t");
  expect_near_vec(to_host(yq), to_host(ref), 1e-4f);
}

TEST(GlimmerQuantDispatch, EmbedDenseAndQuant) {
  const size_t V = 8, D = 16, T = 3;
  Mat table = make_mat(V, D, 100), out("out", T, D), ref("ref", T, D);
  auto toks = make_toks({0, 5, 7});
  KokkosDNN::embedding_gather<Exec, float>(table, toks, ref);
  embed_forward(table, toks, out);
  expect_near_vec(to_host(out), to_host(ref), 0.0f);
  auto packed = pack_f32(V, D, 100);
  Mat outq("outq", T, D);
  embed_forward(packed, toks, outq);
  expect_near_vec(to_host(outq), to_host(ref), 0.0f);
  Mat bad("bad", T, D + 1);
  EXPECT_THROW(embed_forward(packed, toks, bad), std::invalid_argument);
}
