// cuDNN SDPA implementation for KokkosDNN grouped-query causal attention.
//
// Single translation unit that includes <cudnn_frontend.h> so the rest of
// the (header-only) library does not pay the frontend compile cost.
// The header declares cudnn_sdpa_forward() (causal) and
// cudnn_sdpa_forward_noncausal() (dense); here they are either stubs
// returning false (CUDNN disabled or non-CUDA build) or the real graph.
//
// Real path: one inference SDPA node (generate_stats=false). The causal
// variant bakes in bottom-right causal masking
// (diagonal_alignment=BOTTOM_RIGHT, right_bound=0), which matches the
// native kernel's mask when Ktot == q_off + T (checked by the caller);
// the non-causal variant omits the diagonal-mask attributes for dense
// attention. Q/K/V/O are described with
// the caller's Kokkos strides (B=1, dims [B,H,S,D]), so arbitrary
// Kokkos layouts and cache subviews work with no repack copy.
// Implementation selection is AUTO: unified flash for BF16, composite
// (matmul+softmax+matmul) for FP32. Any build/execute failure returns
// false so the caller falls back to the native Kokkos kernel.
#include "KokkosDNN_config.h"

#include <Kokkos_Core.hpp>

#include "tpl/cudnn/KokkosDNN_CuDNN_Forward.hpp"

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cuda_runtime.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

namespace KokkosDNN {
namespace Impl {

namespace {
// Shared SDPA graph builder/executor. causal=true bakes in the
// bottom-right causal mask (right_bound=0); causal=false omits the
// diagonal-mask attributes entirely, which per the cuDNN frontend docs
// yields dense (non-causal) attention.
bool sdpa_forward_impl(void* cached_handle, CudnnSdpaScalar dtype,
                       const void* q, const void* k, const void* v, void* out,
                       int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                       int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                       int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                       int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                       int64_t o_sD, float scale, bool causal) {
  if (cached_handle == nullptr) return false;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr)
    return false;
  if (T <= 0 || Ktot <= 0 || Hq <= 0 || Hkv <= 0 || D <= 0) return false;
  if (Hq % Hkv != 0) return false;
  // Element strides from the Kokkos views must be positive.
  if (q_sT <= 0 || q_sH <= 0 || q_sD <= 0) return false;
  if (k_sT <= 0 || k_sH <= 0 || k_sD <= 0) return false;
  if (v_sT <= 0 || v_sH <= 0 || v_sD <= 0) return false;
  if (o_sT <= 0 || o_sH <= 0 || o_sD <= 0) return false;

  fe::DataType_t io_type = fe::DataType_t::NOT_SET;
  if (dtype == CudnnSdpaScalar::Float) {
    io_type = fe::DataType_t::FLOAT;
  } else if (dtype == CudnnSdpaScalar::BFloat16) {
    io_type = fe::DataType_t::BFLOAT16;
  } else {
    return false;
  }

  // Narrow, version-gated exclusion (cuDNN <= 9.27.0, i.e. CUDNN_VERSION <=
  // 92700): the FP32 composite implementation silently computes only the
  // first query head per KV group at s_q == 1 with GQA (Hq != Hkv) while
  // reporting success (verified: non-leading group heads come back zero).
  // MHA s_q == 1 and GQA s_q >= 2 verified correct to ~1e-7, so only this
  // combination falls back to native. Newer cuDNN releases automatically
  // retry the TPL path here.
#if CUDNN_VERSION <= 92700
  if (dtype == CudnnSdpaScalar::Float && T == 1 && Hq != Hkv) return false;
#endif

  // The caller (Kokkos execution space) owns Q/K/V/O. Fence so prior
  // Kokkos kernels have completed before cuDNN reads them.
  Kokkos::fence();

  // Cached handle owned by the DNNHandle (created once per handle, not per
  // call). Re-select the Kokkos CUDA stream every call so ordering with
  // surrounding Kokkos kernels is preserved without extra synchronization.
  cudnnHandle_t cudnn_handle = static_cast<cudnnHandle_t>(cached_handle);
  const cudaStream_t kokkos_stream = Kokkos::Cuda().cuda_stream();
  if (cudnnSetStream(cudnn_handle, kokkos_stream) != CUDNN_STATUS_SUCCESS)
    return false;

  bool ok = false;
  void* workspace = nullptr;

  try {
    // B=1; cuDNN dims are [B,H,S,D], strides are the caller's Kokkos
    // (token, head, dim) element strides mapped onto (S,H,D). The B
    // stride is unused for B=1; set it to the per-batch footprint.
    const int64_t b = 1;
    const std::vector<int64_t> q_dim = {b, Hq, T, D};
    const std::vector<int64_t> q_stride = {Hq * T * D, q_sH, q_sT, q_sD};
    const std::vector<int64_t> k_dim = {b, Hkv, Ktot, D};
    const std::vector<int64_t> k_stride = {Hkv * Ktot * D, k_sH, k_sT, k_sD};
    const std::vector<int64_t> v_dim = {b, Hkv, Ktot, D};
    const std::vector<int64_t> v_stride = {Hkv * Ktot * D, v_sH, v_sT, v_sD};
    const std::vector<int64_t> o_dim = {b, Hq, T, D};
    const std::vector<int64_t> o_stride = {Hq * T * D, o_sH, o_sT, o_sD};

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim(q_dim)
                               .set_stride(q_stride));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim(k_dim)
                               .set_stride(k_stride));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim(v_dim)
                               .set_stride(v_stride));
    if (Q == nullptr || K == nullptr || V == nullptr) {
      return false;
    }

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name(causal ? "kokkosdnn_sdpa"
                                             : "kokkosdnn_sdpa_noncausal")
                            .set_attn_scale(scale)
                            .set_generate_stats(false);
    if (causal) {
      sdpa_options.set_diagonal_alignment(
          fe::DiagonalAlignment_t::BOTTOM_RIGHT);
      sdpa_options.set_diagonal_band_right_bound(0);
    }
    if (io_type == fe::DataType_t::FLOAT) {
      // FP32 I/O is only served by the composite (matmul + softmax +
      // matmul) implementation; AUTO selection does not route there on
      // all setups, so request it explicitly. BF16 stays on AUTO
      // (unified flash attention).
      sdpa_options.set_implementation(
          fe::AttentionImplementation_t::COMPOSITE);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    if (O == nullptr) {
      return false;
    }
    O->set_output(true).set_dim(o_dim).set_stride(o_stride);

    if (!graph->validate().is_good()) {
      return false;
    }
    if (!graph->build(cudnn_handle,
                      {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
             .is_good()) {
      return false;
    }

    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
      return false;
    }
    if (workspace_size > 0) {
      if (cudaMalloc(&workspace, static_cast<size_t>(workspace_size)) !=
          cudaSuccess) {
        return false;
      }
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{Q, const_cast<void*>(q)},
                        {K, const_cast<void*>(k)},
                        {V, const_cast<void*>(v)},
                        {O, out}};
    ok = graph->execute(cudnn_handle, variant_pack, workspace).is_good();
  } catch (...) {
    ok = false;
  }

  if (workspace != nullptr) cudaFree(workspace);
  // NOTE: cudnn_handle is cached by the DNNHandle -- never destroyed here.
  return ok;
}

}  // namespace

bool cudnn_sdpa_forward(void* cached_handle, CudnnSdpaScalar dtype,
                        const void* q, const void* k, const void* v, void* out,
                        int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                        int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                        int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                        int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                        int64_t o_sD, float scale) {
  return sdpa_forward_impl(cached_handle, dtype, q, k, v, out, T, Ktot, Hq,
                           Hkv, D, q_sT, q_sH, q_sD, k_sT, k_sH, k_sD, v_sT,
                           v_sH, v_sD, o_sT, o_sH, o_sD, scale,
                           /*causal=*/true);
}

bool cudnn_sdpa_forward_noncausal(
    void* cached_handle, CudnnSdpaScalar dtype, const void* q, const void* k,
    const void* v, void* out, int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
    int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD, int64_t k_sT,
    int64_t k_sH, int64_t k_sD, int64_t v_sT, int64_t v_sH, int64_t v_sD,
    int64_t o_sT, int64_t o_sH, int64_t o_sD, float scale) {
  return sdpa_forward_impl(cached_handle, dtype, q, k, v, out, T, Ktot, Hq,
                           Hkv, D, q_sT, q_sH, q_sD, k_sT, k_sH, k_sD, v_sT,
                           v_sH, v_sD, o_sT, o_sH, o_sD, scale,
                           /*causal=*/false);
}

}  // namespace Impl
}  // namespace KokkosDNN

#else

namespace KokkosDNN {
namespace Impl {

bool cudnn_sdpa_forward(void*, CudnnSdpaScalar, const void*, const void*,
                        const void*, void*, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, float) {
  return false;
}

bool cudnn_sdpa_forward_noncausal(void*, CudnnSdpaScalar, const void*,
                                  const void*, const void*, void*, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, float) {
  return false;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
