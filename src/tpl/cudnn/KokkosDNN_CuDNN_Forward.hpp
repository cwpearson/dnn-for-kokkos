#ifndef KOKKOSDNN_CUDNN_FORWARD_HPP
#define KOKKOSDNN_CUDNN_FORWARD_HPP

// Raw-pointer cuDNN SDPA entry point, implemented in
// tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp.
//
// cached_handle is the DNNHandle's cached cudnnHandle_t (opaque void*;
// nullptr declines and the caller falls back to native). The callee
// re-selects the Kokkos CUDA stream on it per call, so the one cached
// handle stays ordered with surrounding Kokkos kernels instead of being
// created/destroyed per call.
// Strides are in elements for (token, head, dim) indexing, i.e. the
// Kokkos View strides: q_sT = q.stride(0), q_sH = q.stride(1), etc.
// Returns true on success, false to fall back to native. Never throws.
#include <cstddef>
#include <cstdint>

#include "tpl/cudnn/KokkosDNN_CuDNN_Types.hpp"

namespace KokkosDNN {
namespace Impl {

bool cudnn_sdpa_forward(void* cached_handle, CudnnSdpaScalar dtype,
                        const void* q, const void* k, const void* v, void* out,
                        int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                        int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                        int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                        int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                        int64_t o_sD, float scale);

}  // namespace Impl
}  // namespace KokkosDNN

#endif
