#ifndef KOKKOSDNN_CUBLAS_HPP
#define KOKKOSDNN_CUBLAS_HPP

// cuBLAS dispatch for KokkosDNN::gemm/gemv. Included into
// src/KokkosDNN_Gemm.hpp (and KokkosDNN_Handle.hpp for the handle helpers)
// only when KOKKOSDNN_ENABLE_CUBLAS is defined. All entry points take the
// caller's cached cublasHandle_t (owned by DNNHandle, created once per
// handle), never creating a handle per call. The stream is (re-)selected
// to the Kokkos CUDA stream on every call so the TPL work stays ordered
// with the surrounding Kokkos kernels.
#include "tpl/cublas/KokkosDNN_CuBLAS_Handle.hpp"
#include "tpl/cublas/KokkosDNN_CuBLAS_Stream.hpp"
#include "tpl/cublas/KokkosDNN_CuBLAS_GemmFloat.hpp"
#include "tpl/cublas/KokkosDNN_CuBLAS_GemmBF16.hpp"

#endif
